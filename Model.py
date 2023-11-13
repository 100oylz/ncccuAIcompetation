from netModule.Backbone import Backbone
from netModule.Neck import Neck
from netModule.Head import Head
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, reg_max: int, num_class: int):
        super(Model, self).__init__()
        self.Backbone = Backbone()
        self.Neck = Neck()
        self.Head = Head(reg_max, num_class)

    def forward(self, x):
        Backbone_out1, Backbone_out2, Backbone_out3 = self.Backbone(x)
        Neck_out1, Neck_out2, Neck_out3 = self.Neck(Backbone_out1, Backbone_out2, Backbone_out3)
        BLS1, CLS1, BLS2, CLS2, BLS3, CLS3 = self.Head(Neck_out1, Neck_out2, Neck_out3)
        return BLS1, CLS1, BLS2, CLS2, BLS3, CLS3