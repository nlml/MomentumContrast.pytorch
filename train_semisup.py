import sys
import gin
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
from network import Net, Net2, MLP, WrapNet


archi_dict = {"Net": Net, "Net2": Net2, "MLP": MLP}

datasets_dict = {
    "mnist": datasets.MNIST,
    "fashion-mnist": datasets.FashionMNIST,
}


@gin.configurable
def get_dataset(name):
    return datasets_dict[name]


def random_subset_of_class_idxs(c):
    where = torch.where(train_mnist_sup.targets == c)[0]
    return rng.choice(where.numpy(), 10)


@gin.configurable
def sup_loader(sup_batch_size, sup_loader_seed):
    if sup_loader_seed > 0:
        assert 0, "test for gin"
    rng = np.random.RandomState(sup_loader_seed)
    sel = np.concatenate(
        [random_subset_of_class_idxs(c) for c in range(10)], 0
    )
    train_mnist_sup.data = train_mnist_sup.data[sel]
    train_mnist_sup.targets = train_mnist_sup.targets[sel]
    return torch.utils.data.DataLoader(
        train_mnist_sup, batch_size=sup_batch_size, shuffle=True
    )


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


@gin.configurable
def momentum_update(model_q, model_k, beta):
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
        k = F.normalize(model_k(x_k), 1)
        k = k.detach()
        queue = queue_data(queue, k)
        queue = dequeue_data(queue, K=10)
        break
    return queue


def _get_p_a_b(a, b):
    # Needed for both walker loss and visit loss
    match_ab = torch.matmul(a, torch.t(b))
    p_ab = F.softmax(match_ab, dim=1)
    return p_ab, match_ab


def calc_walker_loss(
    a, b, p_target, gamma=0.0, visit_weight=0.0, norm=False
):
    if norm:
        a, b = [F.normalize(i, 1) for i in [a, b]]
    p_ab, match_ab = _get_p_a_b(a, b)
    p_ba = F.softmax(match_ab.T, dim=1)
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

        ### Middle calculation ###
        middle = torch.inverse(I - Tbar_uu + 1e-8)
        # middle = I
        # for i in range(1, 2):
        #     middle += torch.matrix_power(Tbar_uu, i)
        # middle /= Tbar_uu.sum(1, keepdim=True)
        # middle = torch.inverse(Tbar_uu + 1e-8)
        # p_aba = torch.matmul(torch.matmul(p_ab, middle), Tbar_ul)
        p_aba = torch.matmul(torch.matmul(p_ab, middle), p_ba)
        ##########################

        p_aba /= p_aba.sum(1, keepdim=True)
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


@gin.configurable
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
    visit_weight=0.0,
    visit_weight_queue=0.0,
    entmin_weight=0.0,
    walk_queue_weight=0.0,
    moco_weight=0.0,
    gamma_queue=0.0,
    norm_logits_to_walker=True,
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
        q, pred_q = model_q(x_q, sup=True, detached=detached)

        if moco_weight:
            with torch.no_grad():
                k = model_k(x_k)
            k = F.normalize(k, 1)

            N = data[0].shape[0]
            K = queue.shape[0]
            l_pos = torch.bmm(F.normalize(q).view(N, 1, -1), k.view(N, -1, 1))
            l_neg = torch.mm(F.normalize(q).view(N, -1), queue.T.view(-1, K))

            logits = torch.cat([l_pos.view(N, 1), l_neg], dim=1)

            labels = torch.zeros(N, dtype=torch.long)
            labels = labels.to(device)
            
            loss_moco = cross_entropy_loss(logits / temp, labels)
            loss += loss_moco * moco_weight
            loss_moco = loss_moco.item()

        if sup_weight > 0.0:
            x_sup, y_sup = get_sup_batch()
            x_sup, y_sup = x_sup.to(device), y_sup.to(device)

            s, pred_sup = model_q(x_sup, sup=True, detached=detached)

            equality_matrix = (y_sup[:, None].eq(y_sup[None, :])).float()
            equality_matrix /= equality_matrix.sum(1, keepdim=True)

            loss_sup = sup_loss_fn(pred_sup, y_sup)
            loss += sup_weight * loss_sup

            if entmin_weight > 0.0:
                loss_entmin = calc_entmin_loss(pred_q)
                loss += entmin_weight * loss_entmin

            if walk_weight > 0.0:
                if walk_queue_weight > 0.0:
                    loss_walker += walk_queue_weight * calc_walker_loss(
                        s,
                        queue,
                        equality_matrix,
                        gamma=gamma_queue,
                        visit_weight=visit_weight_queue,
                        norm=norm_logits_to_walker
                    )
                loss_walker += calc_walker_loss(
                    s,
                    q,
                    equality_matrix,
                    visit_weight=visit_weight,
                    norm=norm_logits_to_walker
                )
                loss += walk_weight * loss_walker

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if moco_weight > 0.0:
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


def go(run_name, batchsize=100, epochs=50, out_dir="result", no_cuda=False):

    gin.parse_config_file(os.path.join("cfg", run_name + ".gin"))

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
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_mnist_sup = get_dataset()(
        "./", train=True, download=True, transform=transform_sup
    )

    train_loader_sup = sup_loader()
    train_loader_sup.iter = iter(train_loader_sup)
    sup_loss_fn = nn.CrossEntropyLoss().cuda()
    cross_entropy_loss = nn.CrossEntropyLoss().cuda()

    use_cuda = not no_cuda and torch.cuda.is_available()
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

    @gin.configurable
    def get_network(archi):
        return WrapNet(archi_dict[archi]())

    model_q = get_network().to(device)
    model_k = get_network().to(device)

    optimizer = optim.SGD(
        model_q.parameters(), lr=0.001, weight_decay=1e-3, momentum=0.9
    )
    queue = initialize_queue(model_k, device, train_loader)

    for epoch in range(1, epochs + 1):
        train(model_q, model_k, device, train_loader, queue, optimizer, epoch)
        test(args, model_q, device, test_loader)

    os.makedirs(out_dir, exist_ok=True)
    torch.save(model_q.state_dict(), os.path.join(out_dir, "model.pth"))

assert len(sys.argv) == 2
go(sys.argv[1])
