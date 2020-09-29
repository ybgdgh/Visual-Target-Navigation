# -*- coding: utf-8 -*-
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

class SemNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SemNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2,
                               bias=False)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.avgpool = nn.AdaptiveMaxPool2d(4)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128*4*4, out_channels)

    def forward(self, x):
        
        # print(x.shape) # 21*128*128
        x = self.conv1(x) # 64*64*64
        # print(x.shape)
        x = self.bn1(x)
        x = F.relu(x, True)
        x = self.maxpool(x) # 64*32*32
        x = self.conv2(x) # 128*16*16
        x = F.relu(x, True)
        # print(x.shape) # 4*128*16*16
        x = self.maxpool(x) # 128*8*8
        # print(x.shape) # 4*128*8*8
        x = self.conv3(x) # 128*4*4
        # print(x.shape) # 4*128*4*4
        x = F.relu(x, True)
        x = x.contiguous().view(x.size(0), -1) # 128*4*4 = 2048
        x = self.fc(x) # out_channels
        # print(x.shape) # 4*512

        return x