# -*- coding: utf-8 -*-
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

class DepNet(nn.Module):
    def __init__(self, out_channels):
        super(DepNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d(4)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128*4*4, out_channels)

    def forward(self, x):
        # 1*128*128
        x = self.conv1(x) # 64*64*64
        x = self.bn1(x)
        x = F.relu(x, True)
        x = self.maxpool(x) # 1*32*32
        x = self.conv2(x) # 128*16*16
        x = F.relu(x, True)
        x = self.avgpool(x) # 128*4*4
        x = x.contiguous().view(x.size(0), -1) # 128*4*4 = 2048
        x = self.fc(x) # out_channels

        return x