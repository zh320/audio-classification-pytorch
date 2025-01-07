"""
Paper:      Deep Residual Learning for Image Recognition
Url:        https://arxiv.org/abs/1512.03385
Create by:  zh320
Date:       2024/07/13
"""

import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


class ResNet(nn.Module):
    # Load ResNet pretrained on ImageNet from torchvision, see
    # https://pytorch.org/vision/stable/models/resnet.html
    def __init__(self, num_class, resnet_type, num_channel=1, pretrained=True, downsample_rate=32):
        super(ResNet, self).__init__()
        resnet_hub = {'resnet18':resnet18, 'resnet34':resnet34, 'resnet50':resnet50,
                        'resnet101':resnet101, 'resnet152':resnet152}
        if resnet_type not in resnet_hub:
            raise ValueError(f'Unsupported ResNet type: {resnet_type}.\n')

        last_channel = 512 if resnet_type in ['resnet18', 'resnet34'] else 2048

        self.model = resnet_hub[resnet_type](pretrained=pretrained)

        if num_channel != 3:
            self.model.conv1 = nn.Conv2d(in_channels=num_channel, 
                                        out_channels=self.model.conv1.out_channels, 
                                        kernel_size=self.model.conv1.kernel_size, 
                                        stride=self.model.conv1.stride, 
                                        padding=self.model.conv1.padding, 
                                        bias=self.model.conv1.bias)

        self.model.fc = nn.Linear(last_channel, num_class)

        if downsample_rate != 32:
            if downsample_rate in [8, 16]:
                self.model.conv1 = nn.Conv2d(3, 64, 7, stride=1, padding=3, bias=False)
                if downsample_rate == 8:
                    self.model.maxpool = nn.Identity()
            else:
                raise NotImplementedError

    def forward(self, x):
        x = self.model(x)

        return x
