"""
Paper:      Look, Listen and Learn
Url:        https://arxiv.org/abs/1705.08168
Create by:  zh320
Date:       2024/12/14
"""

import torch
import torch.nn as nn

from .modules import ConvBNAct, Activation


class L3Net(nn.Module):
    def __init__(self, num_class, num_channel=1, pretrained=False, act_type='relu'):
        super().__init__()
        if pretrained:
            print('No pretrained weight available for L3Net.')

        self.block1 = ConvBlock(num_channel, 64, act_type)
        self.block2 = ConvBlock(64, 128, act_type)
        self.block3 = ConvBlock(128, 256, act_type)
        self.block4 = nn.Sequential(ConvBlock(256, 512, act_type, has_pool=False),
                                    nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Sequential(nn.Linear(512, 128),
                                Activation(act_type))
        self.classifier = nn.Linear(128, num_class)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.classifier(x)

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, act_type, has_pool=True):
        super().__init__()
        layers = [ConvBNAct(in_channel, out_channel, act_type=act_type),
                  ConvBNAct(out_channel, out_channel, act_type=act_type),]
        if has_pool:
            layers.append(nn.MaxPool2d(2,2))

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)