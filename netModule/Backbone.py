import sys

sys.path.append("../")
from netLayer.SPPF import SPPF
from netLayer.DarknetBottleneck import DaeknetBottleneck
from netLayer.CSPLayer_2Conv import CSPLayer_2Conv
from netLayer.ConvModule import ConvModule
from config import yolov8_d, yolov8_r, yolov8_w
import torch
import torch.nn as nn
import torch.nn.functional as F


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.calculate_parameters()
        self.create_layers()

    def calculate_parameters(self):
        """
        计算Backbone中的网络超参数
        在s中，channel分别为：32，64，128，256，512
        csp_n分别为：1，2，2，1
        在m中，channel分别为：48，96，192，384，576
        csp_n分别为：2，4，4，2
        :return: None
        :rtype: None
        """
        self.stem_out_channel = int(round(yolov8_w * 64))
        self.stage1_out_channel = int(round(yolov8_w * 128))
        self.stage2_out_channel = int(round(yolov8_w * 256))
        self.stage3_out_channel = int(round(yolov8_w * 512))
        self.stage4_out_channel = int(round(yolov8_w * 512 * yolov8_r))
        self.stage1_csp_n = int(round(3 * yolov8_d))
        self.stage2_csp_n = int(round(6 * yolov8_d))
        self.stage3_csp_n = int(round(6 * yolov8_d))
        self.stage4_csp_n = int(round(3 * yolov8_d))

    def create_layers(self):
        self.StemLayer = ConvModule(in_channel=3, out_channel=self.stem_out_channel, kernel_size=3, stride=2,
                                    padding=1)
        self.StageLayer1 = nn.Sequential(
            ConvModule(in_channel=self.stem_out_channel, out_channel=self.stage1_out_channel, kernel_size=3, stride=2,
                       padding=1),
            CSPLayer_2Conv(in_channel=self.stage1_out_channel, out_channel=self.stage1_out_channel, add=True,
                           n=self.stage1_csp_n)
        )

        self.StageLayer2 = nn.Sequential(
            ConvModule(in_channel=self.stage1_out_channel, out_channel=self.stage2_out_channel, kernel_size=3, stride=2,
                       padding=1),
            CSPLayer_2Conv(in_channel=self.stage2_out_channel, out_channel=self.stage2_out_channel, add=True,
                           n=self.stage2_csp_n)
        )

        self.StageLayer3 = nn.Sequential(
            ConvModule(in_channel=self.stage2_out_channel, out_channel=self.stage3_out_channel, kernel_size=3, stride=2,
                       padding=1),
            CSPLayer_2Conv(in_channel=self.stage3_out_channel, out_channel=self.stage3_out_channel, add=True,
                           n=self.stage3_csp_n)
        )

        self.StageLayer4 = nn.Sequential(
            ConvModule(in_channel=self.stage3_out_channel, out_channel=self.stage4_out_channel, kernel_size=3, stride=2,
                       padding=1),
            CSPLayer_2Conv(in_channel=self.stage4_out_channel, out_channel=self.stage4_out_channel, add=True,
                           n=self.stage4_csp_n),
            SPPF(in_channel=self.stage4_out_channel, out_channel=self.stage4_out_channel)
        )

    def forward(self, x):
        x = self.StemLayer(x)
        x = self.StageLayer1(x)
        x = self.StageLayer2(x)
        out1 = x
        x = self.StageLayer3(x)
        out2 = x
        x = self.StageLayer4(x)
        out3 = x
        return out1, out2, out3
