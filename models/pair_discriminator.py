import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
# from model import locNN
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math


# 3x3 Convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1,  bias=False)


# Residual Block
class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



# ResNet Module
class ResNet(nn.Module):
    def __init__(self, block, layers, number_class):
        super(ResNet, self).__init__()

        # self.up = F.interpolate(scale_factor=37, mode='bilinear')

        self.conv1 = nn.Sequential(
            nn.Conv2d(3*50*2, 3*50*2, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(3*50*2),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.in_channels = 3*50*2
        self.layer1 = self.make_layer(block, 3*50*2, layers[0]) #8*64*64 -> 16*32*32
        self.layer2 = self.make_layer(block, 3*50*2, layers[1], 2) #16*32*32 -> 32*16*16
        self.layer3 = self.make_layer(block, 512, layers[2], 2) #32*16*16 -> 64*8*8
        self.layer4 = self.make_layer(block, 512, layers[3], 2) #64*8*8 -> 128*4*4

        self.avg = nn.AvgPool2d(kernel_size=7)

        self.fc = nn.Linear(512, number_class)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels*block.expansion):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels*block.expansion, stride=stride),
                nn.BatchNorm2d(out_channels*block.expansion))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

#
    def forward(self, x):
        out = F.interpolate(x, scale_factor=37, mode='bilinear')
        out = self.conv1(out)
        #
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg(out)
        # #
        out = out.squeeze()
        out = self.fc(out)
        # out = self.sm(out)
        return out
