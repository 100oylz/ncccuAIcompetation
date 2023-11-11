import warnings

import torch.nn as nn

from ConvModule import ConvModule


class DaeknetBottleneck(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, add: bool):
        """
        DarknetBottleneck
        :param in_channel: 输入通道数
        :type in_channel: int
        :param out_channel: 输出通道数
        :type out_channel: int
        :param add: 是否进行残差链接
        :type add: int
        """
        super().__init__()
        if (add):
            assert in_channel == out_channel
            if (out_channel % 2 == 1):
                warnings.warn(
                    f"{self.__class__.__name__}，hidden_channel=in_channel//2，in_channel={out_channel}，可能产生舍入误差")
            hidden_channel = in_channel // 2
            self.conv_1 = ConvModule(in_channel=in_channel, out_channel=hidden_channel, kernel_size=3, stride=1,
                                     padding=1)
            self.conv_2 = ConvModule(in_channel=hidden_channel, out_channel=out_channel, kernel_size=3, stride=1,
                                     padding=1)
        else:
            if (out_channel % 2 == 1):
                warnings.warn(
                    f"{self.__class__.__name__}，hidden_channel=out_channel//2，out_channel={out_channel}，可能产生舍入误差")
            hidden_channel = out_channel // 2
            self.conv_1 = ConvModule(in_channel=in_channel, out_channel=hidden_channel, kernel_size=3, stride=1,
                                     padding=1)
            self.conv_2 = ConvModule(in_channel=hidden_channel, out_channel=out_channel, kernel_size=3, stride=1,
                                     padding=1)
        self.add = add

    def forward(self, x):
        if (self.add):
            res = x
            x = self.conv_1(x)
            x = self.conv_2(x)
            return x + res
        else:
            x = self.conv_1(x)
            x = self.conv_2(x)
            return x
