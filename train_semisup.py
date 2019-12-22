from tqdm import tqdm
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

from network import Net, Net2, WrapNet

transform_sup = transforms.Compose(
    [
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(
            28, scale=(0.9, 1.1), ratio=(0.9, 1.1), interpolation=2
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

transform_test = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),]
)


SUP_WEIGHT = 1.0
WALK_WEIGHT = 1.0
VISIT_WEIGHT = 1.0
ENTMIN_WEIGHT = 1.0
SUP_BATCH_SIZE = 100
WALK_QUEUE_WEIGHT = 1.0
MOCO_WEIGHT = 1.0
EMA_BETA = 0.99
GAMMA_QUEUE = 1.0
# DATASET = datasets.MNIST
DATASET = datasets.FashionMNIST
print(
    "SUP_WEIGHT {}, WALK_WEIGHT {}, VISIT_WEIGHT {}, ENTMIN_WEIGHT {}, SUP_BATCH_SIZE {}, WALK_QUEUE_WEIGHT {}, MOCO_WEIGHT {}, EMA_BETA {}, GAMMA_QUEUE {},".format(
        SUP_WEIGHT,
        WALK_WEIGHT,
        VISIT_WEIGHT,
        ENTMIN_WEIGHT,
        SUP_BATCH_SIZE,
        WALK_QUEUE_WEIGHT,
        MOCO_WEIGHT,
        EMA_BETA,
        GAMMA_QUEUE,
    )
)


train_mnist_sup = DATASET(
    "./", train=True, download=True, transform=transform_sup
)
rng = np.random.RandomState(1)


def random_subset_of_class_idxs(c):
    where = torch.where(train_mnist_sup.targets == c)[0]
    return rng.choice(where.numpy(), 10)


sel = np.concatenate([random_subset_of_class_idxs(c) for c in range(10)], 0)
train_mnist_sup.data = train_mnist_sup.data[sel]
train_mnist_sup.targets = train_mnist_sup.targets[sel]
train_loader_sup = torch.utils.data.DataLoader(
    train_mnist_sup, batch_size=SUP_BATCH_SIZE, shuffle=True
)
train_loader_sup.iter = iter(train_loader_sup)
sup_loss_fn = nn.CrossEntropyLoss().cuda()
cross_entropy_loss = nn.CrossEntropyLoss().cuda()


def get_sup_batch():
    try:
        x = next(train_loader_sup.iter)
    except StopIteration:
        train_loader_sup.iter = iter(train_loader_sup)
        x = next(train_loader_sup.iter)
    return x


class DuplicatedCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        img1 = img.copy()
        img2 = img.copy()
        for t in self.transforms:
            img1 = t(img1)
            img2 = t(img2)
        return img1, img2


def momentum_update(model_q, model_k, beta=EMA_BETA):
    param_k = model_k.state_dict()
    param_q = model_q.named_parameters()
    for n, q in param_q:
        if n in param_k:
            param_k[n].data.copy_(beta * param_k[n].data + (1 - beta) * q.data)
    model_k.load_state_dict(param_k)


def queue_data(data, k):
    return torch.cat([data, k], dim=0)


def dequeue_data(data, K=4096):
    if len(data) > K:
        return data[-K:]
    else:
        return data


def initialize_queue(model_k, device, train_loader):
    queue = torch.zeros((0, 128), dtype=torch.float)
    queue = queue.to(device)

    for batch_idx, (data, target) in enumerate(train_loader):
        x_k = data[1]
        x_k = x_k.to(device)
        k = model_k(x_k)
        k = k.detach()
        queue = queue_data(queue, k)
        queue = dequeue_data(queue, K=10)
        break
    return queue


def calc_walker_loss_old(sup_logits, oth_logits, equality_matrix):
    l_sup = torch.mm(sup_logits, oth_logits.T)
    # w_sup contains probs
    p_ab = torch.softmax(l_sup, 1)
    p_ba = torch.softmax(l_sup.T, 1)
    p_aba = torch.mm(p_ab, p_ba)
    # could use multilabel_margin_loss - using cross entropy for now
    # return F.binary_cross_entropy(w_sup, equality_matrix)
    loss = (-equality_matrix * torch.log(p_aba + 1e-8)).sum(1).mean()
    if VISIT_WEIGHT > 0.0:
        loss += calc_visit_loss(p_ab)
    return loss


def _get_p_a_b(a, b):
    # Needed for both walker loss and visit loss
    match_ab = torch.matmul(a, torch.t(b))
    p_ab = F.softmax(match_ab, dim=1)
    return p_ab, match_ab


def calc_walker_loss(a, b, p_target, gamma=0.0):
    p_ab, match_ab = _get_p_a_b(a, b)
    # equality_matrix = (labels.view([-1, 1]).eq(labels)).float()
    # p_target = equality_matrix / equality_matrix.sum(1, keepdim=True)

    if gamma > 0.0:  # Learning by infinite association
        match_ba = torch.t(match_ab)
        match_bb = torch.matmul(b, torch.t(b))
        add = np.log(gamma) if gamma < 1.0 else 0.0
        match_ab_bb = torch.cat([match_ba, match_bb + add], dim=1)
        p_ba_bb = torch.clamp(F.softmax(match_ab_bb, dim=1), min=1e-8)
        N = a.shape[0]
        M = b.shape[0]
        Tbar_ul, Tbar_uu = p_ba_bb[:, :N], p_ba_bb[:, N:]
        I = torch.eye(M)
        I = I.cuda() if Tbar_uu.is_cuda else I
        middle = torch.inverse(I - Tbar_uu + 1e-8)
        p_aba = torch.matmul(torch.matmul(p_ab, middle), Tbar_ul)
    else:  # Original learning by association method
        p_ba = F.softmax(torch.t(match_ab), dim=1)
        p_aba = torch.matmul(p_ab, p_ba)
    loss_aba = -(p_target * torch.log(p_aba + 1e-8)).sum(1).mean(0)
    return loss_aba


def calc_visit_loss(p_ab):
    p_ab_avg = p_ab.mean(0)
    return (p_ab_avg * torch.log(p_ab_avg)).sum()


def calc_entmin_loss(logits):
    p = torch.softmax(logits, 1)
    return -(p * torch.log(p + 1e-8)).sum(1).mean()


def train(
    model_q,
    model_k,
    device,
    train_loader,
    queue,
    optimizer,
    epoch,
    temp=0.07,
    sup_weight=0.0,
    walk_weight=0.0,
    detached=False,
):
    model_q.train()
    total_loss = 0

    print(sup_weight, walk_weight, detached)

    l = np.ceil(len(train_loader.dataset) / train_loader.batch_size)
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=l):

        loss = loss_moco = loss_sup = loss_walker = loss_entmin = 0.0

        x_q = data[0]
        x_k = data[1]

        x_q, x_k = x_q.to(device), x_k.to(device)
        q_pre_norm, pred_q = model_q(x_q, sup=True, detached=detached)
        q = F.normalize(q_pre_norm)

        if MOCO_WEIGHT:
            k = model_k(x_k)
            k = k.detach()

            N = data[0].shape[0]
            K = queue.shape[0]
            l_pos = torch.bmm(q.view(N, 1, -1), k.view(N, -1, 1))
            l_neg = torch.mm(q.view(N, -1), queue.T.view(-1, K))

            logits = torch.cat([l_pos.view(N, 1), l_neg], dim=1)

            labels = torch.zeros(N, dtype=torch.long)
            labels = labels.to(device)

            loss = cross_entropy_loss(logits / temp, labels) * MOCO_WEIGHT
            loss_moco = loss.item()

        if sup_weight > 0.0:
            x_sup, y_sup = get_sup_batch()
            x_sup, y_sup = x_sup.to(device), y_sup.to(device)

            s, pred_sup = model_q(x_sup, sup=True, detached=detached)

            equality_matrix = (y_sup[:, None].eq(y_sup[None, :])).float()
            equality_matrix /= equality_matrix.sum(1, keepdim=True)

            loss_sup = sup_loss_fn(pred_sup, y_sup)
            loss += sup_weight * loss_sup

            if ENTMIN_WEIGHT > 0.0:
                loss_entmin = calc_entmin_loss(pred_q)
                loss += ENTMIN_WEIGHT * loss_entmin

            if walk_weight > 0.0:
                if WALK_QUEUE_WEIGHT > 0.0:
                    loss_walker += WALK_QUEUE_WEIGHT * calc_walker_loss(
                        s, queue, equality_matrix, gamma=GAMMA_QUEUE
                    )
                loss_walker += calc_walker_loss(s, q_pre_norm, equality_matrix)
                loss += walk_weight * sup_weight * loss_walker

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if MOCO_WEIGHT:
            momentum_update(model_q, model_k)

            queue = queue_data(queue, k)
            queue = dequeue_data(queue)

    total_loss /= len(train_loader.dataset)

    print(
        "Train Epoch: {} \tLoss: {:.6f} \tMoco: {:.6f} \tSup {:.6f} \tWalk {:.6f}".format(
            epoch, total_loss, loss_moco, loss_sup, loss_walker
        )
    )


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            s, output = model(data, sup=True)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


parser = argparse.ArgumentParser(description="MoCo example: MNIST")
parser.add_argument(
    "--batchsize",
    "-b",
    type=int,
    default=100,
    help="Number of images in each mini-batch",
)
parser.add_argument(
    "--epochs",
    "-e",
    type=int,
    default=100,
    help="Number of sweeps over the dataset to train",
)
parser.add_argument(
    "--out", "-o", default="result", help="Directory to output the result"
)
parser.add_argument(
    "--no-cuda",
    action="store_true",
    default=False,
    help="disables CUDA training",
)
args = parser.parse_args()

batchsize = args.batchsize
epochs = args.epochs
out_dir = args.out

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {"num_workers": 4, "pin_memory": True}

transform = DuplicatedCompose(
    [
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(
            28, scale=(0.9, 1.1), ratio=(0.9, 1.1), interpolation=2
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

train_mnist = DATASET("./", train=True, download=True, transform=transform)
test_mnist = DATASET(
    "./", train=False, download=True, transform=transform_test
)

train_loader = torch.utils.data.DataLoader(
    train_mnist, batch_size=batchsize, shuffle=True, **kwargs
)
test_loader = torch.utils.data.DataLoader(
    test_mnist, batch_size=batchsize, shuffle=True, **kwargs
)

model_q = WrapNet(Net2()).to(device)
model_k = WrapNet(Net2()).to(device)

# sd = torch.load("pretrained/model.pth")
# model_q.load_state_dict(sd["model"])
# model_k.load_state_dict(sd["model_k"])

optimizer = optim.SGD(
    model_q.parameters(), lr=0.001, weight_decay=1e-3, momentum=0.9
)
queue = initialize_queue(model_k, device, train_loader)

# sup_weight_dict = [10.0] * 40 + np.linspace(0.01, SUP_WEIGHT, 10).tolist() + [SUP_WEIGHT] * 1000
# walk_weight_dict = [0.0] * 40 + np.linspace(0.01, WALK_WEIGHT, 10).tolist() + [WALK_WEIGHT] * 1000
# detached_dict = [True] * 40 + [False] * 1000

sup_weight_dict = [SUP_WEIGHT] * 1000
walk_weight_dict = [WALK_WEIGHT] * 1000
detached_dict = [False] * 1000

# test(args, model_q, device, test_loader)
for epoch in range(1, epochs + 1):
    train(
        model_q,
        model_k,
        device,
        train_loader,
        queue,
        optimizer,
        epoch,
        sup_weight=sup_weight_dict[epoch],
        walk_weight=walk_weight_dict[epoch],
        detached=detached_dict[epoch],
    )
    test(args, model_q, device, test_loader)

os.makedirs(out_dir, exist_ok=True)
torch.save(model_q.state_dict(), os.path.join(out_dir, "model.pth"))
