import torch
import torch.nn as nn
import torch.nn.functional as F


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
