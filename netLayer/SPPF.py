import torch
import torch.nn as nn
import torch.nn.functional as F
from ConvModule import ConvModule
import warnings


class SPPF(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size=5):
        """
        SPPF
        :param in_channel: 输入通道数
        :type in_channel: int
        :param out_channel: 输出通道数
        :type out_channel: int
        :param kernel_size: 最大池化时的kernel_size
        :type kernel_size: int
        """
        super().__init__()
        if (out_channel % 2 == 1):
            warnings.warn(
                f"{self.__class__.__name__}，hidden_channel=in_channel//2，in_channel={out_channel}，可能产生舍入误差")
        hidden_channel = in_channel // 2
        self.conv_1 = ConvModule(
            in_channel=in_channel,
            out_channel=hidden_channel,
            kernel_size=1, stride=1,
            padding=0)
        self.conv_2 = ConvModule(
            in_channel=hidden_channel * 4,
            out_channel=out_channel,
            kernel_size=1, stride=1,
            padding=0)
        self.maxpool2d = nn.MaxPool2d(kernel_size=5, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.conv_1(x)
        y1 = self.maxpool2d(x)
        y2 = self.maxpool2d(y1)
        y3 = self.maxpool2d(y2)
        out = self.conv_2(torch.cat((x, y1, y2, y3), 1))
        return out
