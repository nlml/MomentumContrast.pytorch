import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, layer_sizes = [784, 1000, 500, 250, 250, 128]):
        super(MLP, self).__init__()
        self.layer_sizes = layer_sizes
        mlp_layers = []
        for s_old, s in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            mlp_layers += [
                nn.Linear(s_old, s, bias=True),
                nn.ReLU(),
                nn.BatchNorm1d(s),
                nn.Dropout(0.1)
            ]
        self.drop_inp = nn.Dropout(0.2)
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.drop_inp(x)
        x = self.mlp(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()

        s_old = 1
        s = 32
        self.conv1 = nn.Conv2d(s_old, s, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(s, s, 3, 1, padding=1)
        s_old = s
        s = 64
        self.conv3 = nn.Conv2d(s_old, s, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(s, s, 3, 1, padding=1)
        s_old = s
        s = 128
        self.conv5 = nn.Conv2d(s_old, s, 3, 1, padding=1)
        self.conv6 = nn.Conv2d(s, s, 3, 1, padding=1)
        s_old = s

        self.fc1 = nn.Linear(1152, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = F.elu(x)
        x = self.conv2(x)
        x = F.elu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = F.elu(x)
        x = self.conv4(x)
        x = F.elu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv5(x)
        x = F.elu(x)
        x = self.conv6(x)
        x = F.elu(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
        # return F.normalize(x)


class Net(nn.Module):
    def __init__(self, sup_out=False):
        super(Net, self).__init__()
        self.sup_out = sup_out
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        if self.sup_out:
            self.fc_sup = nn.Linear(128, 10)

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
    def __init__(self, features):
        super(WrapNet, self).__init__()
        self.features = features
        self.fc_sup = nn.Linear(128, 10)

    def forward(self, x, sup=False, detached=False):
        if detached:
            with torch.no_grad():
                x = self.features(x)
        else:
            x = self.features(x)
        if sup:
            return x, self.fc_sup(x)
        return x
