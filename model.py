import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in')
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Sequential(
            Conv2d(1, 32, kernel_size=3),
            Conv2d(32, 64, kernel_size=3),
            Conv2d(64, 128, kernel_size=3),
            nn.MaxPool2d(2),
            Conv2d(128, 256, kernel_size=3),
            Conv2d(256, 512, kernel_size=3),
            Conv2d(512, 1024, kernel_size=3),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
