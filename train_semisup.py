import time
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
import pandas as pd

sup_loss_fn = nn.CrossEntropyLoss()
cross_entropy_loss = nn.CrossEntropyLoss()

archi_dict = {"Net": Net, "Net2": Net2, "MLP": MLP}

datasets_dict = {
    "mnist": datasets.MNIST,
    "fashion-mnist": datasets.FashionMNIST,
}

train_cols = ["loss", "loss_moco", "loss_sup", "loss_entmin", "loss_walker"]
valid_cols = ["loss", "accuracy"]


def get_transform_sup():
    return transforms.Compose(
        [
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(
                28, scale=(0.9, 1.1), ratio=(0.9, 1.1), interpolation=2
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


@gin.configurable
def get_dataset(name):
    return datasets_dict[name]


@gin.configurable
def get_network(archi):
    return WrapNet(archi_dict[archi]())


def random_subset_of_class_idxs(rng, targets, c):
    where = torch.where(targets == c)[0]
    return rng.choice(where.numpy(), 10)


@gin.configurable
def sup_loader(sup_batch_size, sup_loader_seed):
    train_mnist_sup = get_dataset()(
        "./", train=True, download=True, transform=get_transform_sup()
    )
    rng = np.random.RandomState(sup_loader_seed)
    targs = train_mnist_sup.targets
    sel = np.concatenate(
        [random_subset_of_class_idxs(rng, targs, c) for c in range(10)], 0
    )
    train_mnist_sup.data = train_mnist_sup.data[sel]
    train_mnist_sup.targets = train_mnist_sup.targets[sel]
    return torch.utils.data.DataLoader(
        train_mnist_sup, batch_size=sup_batch_size, shuffle=True
    )


def get_sup_batch(train_loader_sup):
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
    a, b, p_target, gamma=0.0, visit_weight=0.0, norm=False, temp=1.0, inv_lim=1024
):
    if gamma > 0 and b.shape[0] > inv_lim:
        b = b[torch.randperm(b.shape[0])[:inv_lim]]
    if norm:
        a, b = [F.normalize(i, 1) for i in [a, b]]
    p_ab, match_ab = _get_p_a_b(a, b)
    p_ba = F.softmax(match_ab.T / temp, dim=1)
    # equality_matrix = (labels.view([-1, 1]).eq(labels)).float()
    # p_target = equality_matrix / equality_matrix.sum(1, keepdim=True)

    if gamma > 0.0:  # Learning by infinite association
        match_ba = torch.t(match_ab)
        match_bb = torch.matmul(b, torch.t(b))

        add = np.log(gamma) if gamma < 1.0 else 0.0
        match_ab_bb = torch.cat([match_ba, match_bb + add], dim=1)
        p_ba_bb = torch.clamp(F.softmax(match_ab_bb / temp, dim=1), min=1e-8)
        N = a.shape[0]
        M = b.shape[0]
        Tbar_ul, Tbar_uu = p_ba_bb[:, :N], p_ba_bb[:, N:]
        I = torch.eye(M)
        I = I.cuda() if Tbar_uu.is_cuda else I

        ### Middle calculation ###
        with torch.no_grad():
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
        p_ba = F.softmax(torch.t(match_ab) / temp, dim=1)
        p_aba = torch.matmul(p_ab, p_ba)
    loss_aba = -(p_target * torch.log(p_aba + 1e-8)).sum(1).mean(0)
    return loss_aba


def calc_visit_loss(p_ab):
    p_ab_avg = p_ab.mean(0)
    return (p_ab_avg * torch.log(p_ab_avg)).sum()


def calc_entmin_loss(logits):
    p = torch.softmax(logits, 1)
    return -(p * torch.log(p + 1e-8)).sum(1).mean()


def update_with_metrics(dfs, train_or_valid, run_name, metrics, epoch):
    df = dfs[train_or_valid]
    df = pd.concat(
        [df, pd.DataFrame(metrics[None, :], columns=df.columns, index=[epoch])]
    )
    save_path = os.path.join("logs", run_name)
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(os.path.join(save_path, f"{train_or_valid}.csv"))
    dfs[train_or_valid] = df
    return dfs


@gin.configurable
def train(
    dfs,
    run_name,
    model_q,
    model_k,
    device,
    train_loader,
    train_loader_sup,
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
    walker_temp=1.0,
    normalise_queue_to_walker=False,
):
    model_q.train()
    (
        total_loss,
        total_loss_moco,
        total_loss_sup,
        total_loss_entmin,
        total_loss_walker,
    ) = [0] * 5

    l = np.ceil(len(train_loader.dataset) / train_loader.batch_size)
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=l):

        loss = (
            loss_moco
        ) = loss_sup = loss_walker = loss_entmin = torch.FloatTensor([0.0]).to(
            device
        )

        x_q = data[0]
        x_k = data[1]

        x_q, x_k = x_q.to(device), x_k.to(device)
        q, pred_q = model_q(x_q, sup=True, detached=detached)

        if moco_weight > 0.0:
            with torch.no_grad():
                k = model_k(x_k)

            N = data[0].shape[0]
            K = queue.shape[0]
            l_pos = torch.bmm(F.normalize(q).view(N, 1, -1), k.view(N, -1, 1))
            l_neg = torch.mm(F.normalize(q).view(N, -1), queue.T.view(-1, K))

            logits = torch.cat([l_pos.view(N, 1), l_neg], dim=1)

            labels = torch.zeros(N, dtype=torch.long)
            labels = labels.to(device)

            loss_moco = cross_entropy_loss(logits / temp, labels)
            loss += loss_moco * moco_weight

        if sup_weight > 0.0:
            x_sup, y_sup = get_sup_batch(train_loader_sup)
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
                    assert moco_weight > 0, "need moco for queue walker loss"
                    loss_walker += walk_queue_weight * calc_walker_loss(
                        s,
                        F.normalize(queue, 1)
                        if normalise_queue_to_walker
                        else queue,
                        equality_matrix,
                        gamma=gamma_queue,
                        visit_weight=visit_weight_queue,
                        norm=norm_logits_to_walker,
                        temp=walker_temp,
                    )
                loss_walker += calc_walker_loss(
                    s,
                    q,
                    equality_matrix,
                    visit_weight=visit_weight,
                    norm=norm_logits_to_walker,
                    temp=walker_temp,
                )
                loss += walk_weight * loss_walker

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_loss_moco += loss_moco.item()
        total_loss_sup += loss_sup.item()
        total_loss_entmin += loss_entmin.item()
        total_loss_walker += loss_walker.item()

        if moco_weight > 0.0:
            momentum_update(model_q, model_k)

            queue = queue_data(queue, k)
            queue = dequeue_data(queue)

    metrics = np.array(
        [
            i / len(train_loader.dataset)
            for i in [
                total_loss,
                total_loss_moco,
                total_loss_sup,
                total_loss_entmin,
                total_loss_walker,
            ]
        ]
    )

    print(
        "Train Epoch: {} | Loss: {:.6f} | Moco: {:.6f} | Sup {:.6f} | Entmin {:.6f} | Walk {:.6f}".format(
            epoch, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4]
        )
    )
    return update_with_metrics(dfs, "train", run_name, metrics, epoch)


def test(dfs, model, epoch, device, test_loader, run_name):
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
    test_accu = correct / len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * test_accu
        )
    )
    metrics = np.array([test_loss, test_accu])
    return update_with_metrics(dfs, "valid", run_name, metrics, epoch)


@gin.configurable
def get_args(batchsize=100, epochs=50, out_dir="result", no_cuda=False):
    return batchsize, epochs, out_dir, no_cuda


def go(run_name):
    if run_name.startswith("cfg/"):
        run_name = run_name[4:]
    if run_name.endswith(".gin"):
        run_name = run_name[:-4]
    print(run_name)
    cfg_path = os.path.join("cfg", run_name + ".gin")
    gin.parse_config_file(cfg_path)
    print("Parsed config:")
    print(cfg_path)

    batchsize, epochs, out_dir, no_cuda = get_args()

    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_loader_sup = sup_loader()
    train_loader_sup.iter = iter(train_loader_sup)

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

    train_mnist = get_dataset()(
        "./", train=True, download=True, transform=transform
    )
    test_mnist = get_dataset()(
        "./", train=False, download=True, transform=transform_test
    )

    train_loader = torch.utils.data.DataLoader(
        train_mnist, batch_size=batchsize, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_mnist, batch_size=batchsize, shuffle=True, **kwargs
    )

    model_q = get_network().to(device)
    model_k = get_network().to(device)

    optimizer = optim.SGD(
        model_q.parameters(), lr=0.001, weight_decay=1e-3, momentum=0.9
    )
    queue = initialize_queue(model_k, device, train_loader)

    dfs = {
        "train": pd.DataFrame(
            np.ones([0, len(train_cols)]), columns=train_cols
        ),
        "valid": pd.DataFrame(
            np.ones([0, len(valid_cols)]), columns=valid_cols
        ),
    }

    for epoch in range(1, epochs + 1):
        dfs = train(
            dfs,
            run_name,
            model_q,
            model_k,
            device,
            train_loader,
            train_loader_sup,
            queue,
            optimizer,
            epoch,
        )
        dfs = test(dfs, model_q, epoch, device, test_loader, run_name)

    # os.makedirs(out_dir, exist_ok=True)
    # torch.save(model_q.state_dict(), os.path.join(out_dir, "model.pth"))


assert len(sys.argv) == 2
go(sys.argv[1])
