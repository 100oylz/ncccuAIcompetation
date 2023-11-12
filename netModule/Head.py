from netLayer.ConvModule import ConvModule
import torch
import torch.nn as nn

from config import neck_upsample_mode, yolov8_d, yolov8_r, yolov8_w

class Head(nn.Module):
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
        self.convmodule_1 = ConvModule(in_channel=self.channel_256_w, out_channel=self.channel_256_w, kernel_size=3,
                                       stride=1, padding=1)
        self.convmodule_2 = ConvModule(in_channel=self.channel_512_w, out_channel=self.channel_512_w, kernel_size=3,
                                       stride=1, padding=1)
        self.convmodule_3 = ConvModule(in_channel=self.channel_512_w_r, out_channel=self.channel_512_w_r, kernel_size=3,
                                       stride=1, padding=1)

        self.conv1 = nn.Conv2d(kernel_size=1,stride=1,padding=0,in_channels=self.channel_256_w,out_channels=4*100)
        self.conv2 = nn.Conv2d(kernel_size=1,stride=1,padding=0,in_channels=self.channel_256_w,out_channels=9)
        self.conv3 = nn.Conv2d(kernel_size=1,stride=1,padding=0,in_channels=self.channel_512_w,out_channels=4*100)
        self.conv4 = nn.Conv2d(kernel_size=1,stride=1,padding=0,in_channels=self.channel_512_w,out_channels=9)
        self.conv5 = nn.Conv2d(kernel_size=1, stride=1, padding=0, in_channels=self.channel_512_w_r, out_channels=4 * 100)
        self.conv6 = nn.Conv2d(kernel_size=1, stride=1, padding=0, in_channels=self.channel_512_w_r, out_channels=9)

    def forward(self,x1,x2,x3):
        out1 = self.convmodule_1(x1)
        BLS1 = self.conv1(out1)
        CLS1 = self.conv2(out1)

        out2 = self.convmodule_2(x2)
        BLS2 = self.conv3(out2)
        CLS2 = self.conv4(out2)

        out3 = self.convmodule_3(x3)
        BLS3 = self.conv5(out3)
        CLS3 = self.conv6(out3)

        return BLS1,CLS1,BLS2,CLS2,BLS3,CLS3
