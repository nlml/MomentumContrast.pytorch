import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from network import Net, WrapNet
import torch
import torch.utils.data as tud
from sklearn.linear_model import LogisticRegression


def get_all_x(mnist):
    loader = tud.DataLoader(mnist, batch_size=100, num_workers=8)
    all_x = []
    for x, y in tqdm.tqdm(loader):
        with torch.no_grad():
            all_x.append(model(x.to(device)).cpu())
    return torch.cat(all_x)


def show(mnist, targets, ret):
    target_ids = range(len(set(targets)))

    colors = ["r", "g", "b", "c", "m", "y", "k", "violet", "orange", "purple"]

    plt.figure(figsize=(12, 10))

    ax = plt.subplot(aspect="equal")
    for label in set(targets):
        idx = np.where(np.array(targets) == label)[0]
        plt.scatter(ret[idx, 0], ret[idx, 1], c=colors[label], label=label)

    for i in range(0, len(targets), 250):
        img = (mnist[i][0] * 0.3081 + 0.1307).numpy()[0]
        img = OffsetImage(img, cmap=plt.cm.gray_r, zoom=0.5)
        ax.add_artist(AnnotationBbox(img, ret[i]))

    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoCo example: MNIST")
    parser.add_argument(
        "--model", "-m", default="pretrained/model.pth", help="Model file"
    )
    args = parser.parse_args()
    model_path = args.model

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    mnist = datasets.MNIST(
        "./", train=True, download=True, transform=transform
    )
    if 1:  # just 100 labels
        rng = np.random.RandomState(1)
        sel = np.concatenate([rng.choice(torch.where(mnist.targets == c)[0].numpy(), 10) for c in range(10)], 0)
        mnist.data = mnist.data[sel]
        mnist.targets = mnist.targets[sel]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net().to(device)
    sd = torch.load(model_path, map_location=torch.device("cpu"))["model"]
    model.load_state_dict(sd)
    model = WrapNet(model)
    mnist_test = datasets.MNIST(
        "./", train=False, download=True, transform=transform
    )
    all_x = get_all_x(mnist)
    all_x_test = get_all_x(mnist_test)
    for C in np.logspace(2, 9, 5):
        print()
        print("C =", C)
        lr = LogisticRegression(
            C=C, solver="lbfgs", multi_class="multinomial", max_iter=500
        )
        lr.fit(all_x, mnist.targets)
        print((lr.predict(all_x_test) == mnist_test.targets.numpy()).mean())
