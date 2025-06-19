import torch
import torch.nn as nn
from src.core import register
import math
import time

# Channel-Separated Networks

@register
class CSN(torch.nn.Module): # 0.94, 0.5185, FPS: 105
    def __init__(self, feat_strides=[8, 16, 32], reshape=False, model_name='csn_r101', cdim=[96, 256, 1024]):
        super(CSN, self).__init__()
        self.model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)

        self.AdaptiveAvgPool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.model = self.model.blocks[:int(math.log(feat_strides[-1], 2))]
        self.num_levels = len(feat_strides)
        self.reshape = reshape

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = []
        for i in range(len(self.model)):
            x = self.model[i](x)
            if i >= len(self.model) - self.num_levels:
                out.append(x.view(x.size(0), -1, x.size(3), x.size(4)) if self.reshape else x)
        act = self.AdaptiveAvgPool3d(out[-1]).view(x.size(0), -1)

        return out, act

if __name__ == '__main__':
    model = CSN(feat_strides=[8, 16, 32])
    model.eval()
    # model = model.cuda()
    x = torch.rand(1, 3, 9, 640, 640)# .cuda()

    out, act = model(x)
    print([o.shape for o in out])
    print(act.shape)