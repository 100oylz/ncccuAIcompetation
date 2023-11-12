from netLayer.ConvModule import ConvModule
from netLayer.CSPLayer_2Conv import CSPLayer_2Conv
import torch
import torch.nn as nn

from config import neck_upsample_mode, yolov8_d, yolov8_r, yolov8_w


class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.calculateParameters()
        self.createLayer()

    def calculateParameters(self):
        self.channel_512_w_r = int(round(512 * yolov8_w * yolov8_r))
        self.channel_512_w_r_1 = int(round(512 * yolov8_w * (yolov8_r + 1)))
        self.channel_256_w = int(round(256 * yolov8_w))
        self.channel_512_w = int(round(512 * yolov8_w))
        self.csp_num = int(round(3 * yolov8_d))

    def createLayer(self):
        self.up_sample = nn.Upsample(scale_factor=2, mode=neck_upsample_mode)
        self.csplayer_2conv_1 = CSPLayer_2Conv(in_channel=self.channel_512_w_r_1, out_channel=self.channel_512_w,
                                               add=False, n=self.csp_num)
        self.csplayer_2conv_2 = CSPLayer_2Conv(in_channel=self.channel_512_w, out_channel=self.channel_256_w,
                                               add=False, n=self.csp_num)
        self.csplayer_2conv_3 = CSPLayer_2Conv(in_channel=self.channel_512_w, out_channel=self.channel_512_w,
                                               add=False, n=self.csp_num)
        self.csplayer_2conv_4 = CSPLayer_2Conv(in_channel=self.channel_512_w_r_1, out_channel=self.channel_512_w_r,
                                               add=False, n=self.csp_num)
        self.convmodule_1 = ConvModule(in_channel=self.channel_256_w, out_channel=self.channel_256_w, kernel_size=3,
                                       stride=2, padding=1)
        self.convmodule_2 = ConvModule(in_channel=self.channel_512_w, out_channel=self.channel_512_w, kernel_size=3,
                                       stride=2, padding=1)

    def forward(self, x1, x2, x3):
        x3_upsample = self.up_sample(x3)
        x3_upsample = torch.cat((x2, x3_upsample), dim=1)
        # 第一个CSPLayer计算
        x3_upsample = self.csplayer_2conv_1(x3_upsample)

        x2_upsample = self.up_sample(x3_upsample)
        x2_upsample = torch.cat((x1, x2_upsample), dim=1)
        # out1 80*80*256*w 两个Upsample，两个cat，两个CSPLayer
        out1 = self.csplayer_2conv_2(x2_upsample)

        x2_upsample = self.convmodule_1(out1)
        x2_upsample = torch.cat((x2_upsample, x3_upsample), dim=1)
        # out2 40*40*512*w 2个Upsample，3个cat，3个CSPLayer，1个ConvModule
        out2 = self.csplayer_2conv_3(x2_upsample)

        x2_upsample = self.csplayer_2conv_3(out2)

        x1_upsample = torch.cat((x3, x2_upsample), dim=1)
        # out3 20*20*512*w*（r+1） 拼接out2与输入x3
        out3 = self.csplayer_2conv_4(x1_upsample)

        return out1, out2, out3
