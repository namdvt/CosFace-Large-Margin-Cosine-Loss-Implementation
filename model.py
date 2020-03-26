import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.linear(x)
        if x.shape[0] != 1:
            x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Sequential(
            Conv2d(1, 16, kernel_size=3),
            Conv2d(16, 16, kernel_size=3),
            Conv2d(16, 16, kernel_size=3),
            nn.MaxPool2d(2),
            Conv2d(16, 32, kernel_size=3),
            Conv2d(32, 32, kernel_size=3),
            Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            Linear(128, 64),
            Linear(64, 32),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
