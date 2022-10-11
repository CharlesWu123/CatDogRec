# -*- coding: utf-8 -*-
'''
@Version : 0.1
@Author : Charles
@Time : 2022/10/7 14:46
@File : vgg_own.py
@Desc :
'''
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, is_bn=False):
        super(BasicBlock, self).__init__()
        self.is_bn = is_bn
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        if is_bn:
            self.bn = nn.BatchNorm2d(out_channel)       # 论文中没有 bn 层
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.is_bn:
            x = self.bn(x)
        x = self.relu(x)
        return x


class VGG(nn.Module):
    def __init__(self, block, layers, in_channels=3, num_classes=1000, is_dropout=False, is_bn=False):
        super(VGG, self).__init__()
        self.is_dropout = is_dropout
        self.is_bn = is_bn
        self.layer1 = self._make_layer(block, in_channels, 64, layers[0])
        self.layer2 = self._make_layer(block, 64, 128, layers[1])
        self.layer3 = self._make_layer(block, 128, 256, layers[2])
        self.layer4 = self._make_layer(block, 256, 512, layers[3])
        self.layer5 = self._make_layer(block, 512, 512, layers[4])

        self.fc1 = nn.Linear(7*7*512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.relu = nn.ReLU()
        if self.is_dropout:
            self.dropout = nn.Dropout(p=0.5)
        self._init_weight()

    def _make_layer(self, block, in_channel, out_channel, block_num):
        layers = []
        for i in range(block_num):
            if i == 0:
                layers.append(block(in_channel, out_channel, is_bn=self.is_bn))
            else:
                layers.append(block(out_channel, out_channel, is_bn=self.is_bn))
        layers.append(nn.MaxPool2d(2, 2))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        if self.is_dropout:
            x = self.dropout(x)
        x = self.relu(self.fc2(x))
        if self.is_dropout:
            x = self.dropout(x)
        x = self.fc3(x)
        probs = torch.softmax(x, dim=1)
        return x, probs


def vgg19(num_classes=1000, is_dropout=False, is_bn=False):
    model = VGG(BasicBlock, [2, 2, 4, 4, 4], num_classes=num_classes, is_dropout=is_dropout, is_bn=is_bn)
    return model


def vgg16(num_classes=1000, is_dropout=False, is_bn=False):
    model = VGG(BasicBlock, [2, 2, 3, 3, 3], num_classes=num_classes, is_dropout=is_dropout, is_bn=is_bn)
    return model


if __name__ == '__main__':
    import torch

    x = torch.zeros(2, 3, 64, 64)
    net = vgg19()
    y, probs = net(x)
    print(net)
    print(y.shape)
