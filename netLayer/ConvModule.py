import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int, stride: int, padding: int):
        """
        定义了ConvModule
        :param kernel_size: 卷积核大小
        :type kernel_size: int
        :param stride: 步长
        :type stride: int
        :param padding: 填充宽度
        :type padding: int
        :param in_channel: 输入通道数
        :type in_channel: int
        :param out_channel: 输出通道数
        :type out_channel: int
        """
        super().__init__()
        self.conv_module_net = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(num_features=out_channel),
            nn.SiLU(inplace=True)
            # nn.ReLU()
        )

    def forward(self, x):
        return self.conv_module_net(x)
