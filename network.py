import torch
import torch.nn as nn
import torch.nn.functional as F
from bn_multigpuemulate import (
    BatchNorm1dEmulateMultiGPU,
    BatchNorm2dEmulateMultiGPU,
)


class MLP(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        use_bn=True,
        layer_sizes=[784, 1000, 500, 250, 250],
        num_gpus_to_emulate_bn=0,
        bn_before_relu=False,
    ):
        super(MLP, self).__init__()
        self.use_bn = use_bn
        self.bn_before_relu = bn_before_relu
        self.num_gpus_to_emulate_bn = num_gpus_to_emulate_bn
        self.latent_dim = latent_dim
        self.layer_sizes = layer_sizes + [latent_dim]
        mlp_layers = []
        for s_old, s in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            mlp_layers += [nn.Linear(s_old, s, bias=True)]
            if not self.bn_before_relu:
                mlp_layers += [nn.ReLU()]
            if self.use_bn:
                if self.num_gpus_to_emulate_bn:
                    mlp_layers += [
                        BatchNorm1dEmulateMultiGPU(s, num_gpus_to_emulate_bn)
                    ]
                else:
                    mlp_layers += [nn.BatchNorm1d(s)]
            if self.bn_before_relu:
                mlp_layers += [nn.ReLU()]
            mlp_layers += [nn.Dropout(0.1)]
        self.drop_inp = nn.Dropout(0.2)
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.drop_inp(x)
        x = self.mlp(x)
        return x


def get_conv(
    s_in,
    s_out,
    k,
    stride,
    padding,
    use_bn=False,
    num_gpus_to_emulate_bn=0,
    bn_before_relu=False,
):
    out = [nn.Conv2d(s_in, s_out, k, stride, padding=padding)]
    if not bn_before_relu:
        out += [nn.ELU()]
    if use_bn:
        if num_gpus_to_emulate_bn:
            out += [BatchNorm2dEmulateMultiGPU(s_out, num_gpus_to_emulate_bn)]
        else:
            out += [nn.BatchNorm2d(s_out)]
    if bn_before_relu:
        out += [nn.ELU()]
    return nn.Sequential(*out)


class Net2(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        use_bn=False,
        num_gpus_to_emulate_bn=0,
        bn_before_relu=False,
    ):
        super(Net2, self).__init__()
        self.latent_dim = latent_dim
        self.num_gpus_to_emulate_bn = num_gpus_to_emulate_bn
        self.use_bn = use_bn
        self.bn_before_relu = bn_before_relu

        s_old = 1
        s = 32
        self.conv1 = get_conv(
            s_old,
            s,
            3,
            1,
            1,
            self.use_bn,
            self.num_gpus_to_emulate_bn,
            self.bn_before_relu,
        )
        self.conv2 = get_conv(
            s,
            s,
            3,
            1,
            1,
            self.use_bn,
            self.num_gpus_to_emulate_bn,
            self.bn_before_relu,
        )
        s_old = s
        s = 64
        self.conv3 = get_conv(
            s_old,
            s,
            3,
            1,
            1,
            self.use_bn,
            self.num_gpus_to_emulate_bn,
            self.bn_before_relu,
        )
        self.conv4 = get_conv(
            s,
            s,
            3,
            1,
            1,
            self.use_bn,
            self.num_gpus_to_emulate_bn,
            self.bn_before_relu,
        )
        s_old = s
        s = 128
        self.conv5 = get_conv(
            s_old,
            s,
            3,
            1,
            1,
            self.use_bn,
            self.num_gpus_to_emulate_bn,
            self.bn_before_relu,
        )
        self.conv6 = get_conv(
            s,
            s,
            3,
            1,
            1,
            self.use_bn,
            self.num_gpus_to_emulate_bn,
            self.bn_before_relu,
        )
        s_old = s

        self.fc1 = nn.Linear(1152, self.latent_dim)
        self.bn1 = nn.BatchNorm1d(self.latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)

        x = self.conv5(x)
        x = self.conv6(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        return x
        # return F.normalize(x)


class Net(nn.Module):
    def __init__(self, sup_out=False, latent_dim=128):
        super(Net, self).__init__()
        self.latent_dim = latent_dim
        self.sup_out = sup_out
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, self.latent_dim)
        if self.sup_out:
            self.fc_sup = nn.Linear(self.latent_dim, 10)

    def features(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return F.normalize(x)

    def forward(self, x, sup=False, detached=False):
        if detached:
            with torch.no_grad():
                x = self.features(x)
        else:
            x = self.features(x)
        if sup:
            return x, self.fc_sup(x)
        return x


class WrapNet(nn.Module):
    def __init__(self, features, latent_dim=None):
        super(WrapNet, self).__init__()
        self.features = features
        self.latent_dim = (
            self.features.latent_dim if latent_dim is None else latent_dim
        )
        self.fc_sup = nn.Linear(self.latent_dim, 10)

    def forward(self, x, sup=False, detached=False):
        if detached:
            with torch.no_grad():
                x = self.features(x)
        else:
            x = self.features(x)
        if sup:
            return x, self.fc_sup(x)
        return x
