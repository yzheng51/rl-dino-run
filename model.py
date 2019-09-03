import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, batch_norm=False):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(64)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x.view(x.size(0), -1)


class Net(ConvNet):
    """Neural network to estimate Q-value"""
    def __init__(self, n_actions, batch_norm=False):
        super().__init__(batch_norm)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        x = super().forward(x)
        x = F.relu(self.fc(x))
        return self.head(x)


class DuelNet(ConvNet):
    def __init__(self, n_actions, batch_norm=False):
        super().__init__(batch_norm)
        self.fc_a = nn.Linear(3136, 512)
        self.fc_v = nn.Linear(3136, 512)
        self.head_a = nn.Linear(512, n_actions)
        self.head_v = nn.Linear(512, 1)

    def forward(self, x):
        x = super().forward(x)
        x_a = F.relu(self.fc_a(x))
        x_v = F.relu(self.fc_v(x))

        x_a = self.head_a(x_a)
        x_v = self.head_v(x_v).expand_as(x_a)
        return x_v + x_a - x_a.mean(1).unsqueeze(1).expand_as(x_a)
