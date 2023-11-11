from ConvModule import ConvModule
from DarkNetBottleneck import DaeknetBottleneck
import torch.nn as nn
import torch
import warnings


class CSPLayer_2Conv(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, add: bool, n: int):
        """
        CSPLayer_2Conv
        :param in_channel: 输入通道数
        :type in_channel: int
        :param out_channel: 输出通道数
        :type out_channel: int
        :param add: 是否进行残差链接
        :type add: bool
        :param n: n层darknet
        :type n: int
        """
        super().__init__()
        if (out_channel % 2 == 1):
            warnings.warn("hidden_channel=out_channel//2,out_channel可能产生舍入误差", UserWarning)
        self.hidden_channel = out_channel // 2
        self.conv_1 = ConvModule(in_channel=in_channel, out_channel=out_channel, kernel_size=1, padding=0, stride=1)
        self.conv_2 = ConvModule(in_channel=(n + 2) * self.hidden_channel, out_channel=out_channel, kernel_size=1,
                                 stride=1, padding=0)
        self.darknetlist = nn.ModuleList(
            n * [DaeknetBottleneck(in_channel=self.hidden_channel, out_channel=self.hidden_channel, add=add)]
        )

    def forward(self, x):
        y = list(self.conv_1(x).split((self.hidden_channel, self.hidden_channel), 1))
        y.extend(darknet(y[-1]) for darknet in self.darknetlist)
        out = self.conv_2(torch.cat(y, dim=1))
        return out
