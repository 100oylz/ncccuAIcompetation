import torch.nn as nn
import torch.nn.functional as F
from ConvModule import ConvModule


class DaeknetBottleneck(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, add: bool):
        super().__init__()
        if (add):
            assert in_channel == out_channel
            hidden_channel = in_channel // 2
            self.conv_1 = nn.Conv2d(in_channel, hidden_channel, kernel_size=3, stride=1, padding=1)
            self.conv_2 = nn.Conv2d(hidden_channel, out_channel, kernel_size=3, stride=1, padding=1)
        else:
            hidden_channel = out_channel // 2
            self.conv_1 = nn.Conv2d(in_channel, hidden_channel, kernel_size=3, stride=1, padding=1)
            self.conv_2 = nn.Conv2d(hidden_channel, out_channel, kernel_size=3, stride=1, padding=1)
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
