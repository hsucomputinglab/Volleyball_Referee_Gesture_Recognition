import torch
import torch.nn as nn
from src.core import register
import math
import time


@register
class X3D_xs(torch.nn.Module):
    def __init__(self, feat_strides=[8, 16, 32], model_name='x3d_xs', cdim=[96, 256, 1024]):
        super(X3D_xs, self).__init__()
        self.model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
        # -----------------------------------------------------------------------------------------
        self.classifier = self.model.blocks[-1].pool
        self.classifier.pre_conv = nn.Conv3d(cdim[0], cdim[1], kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.classifier.pre_norm = nn.BatchNorm3d(cdim[1])
        self.classifier.post_conv = nn.Conv3d(cdim[1], cdim[2], kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        # -----------------------------------------------------------------------------------------
        self.AdaptiveAvgPool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.model = self.model.blocks[:int(math.log(feat_strides[-1], 2))]
        self.num_levels = len(feat_strides)


    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = []
        for i in range(len(self.model)):
            x = self.model[i](x)
            if i >= len(self.model) - self.num_levels:
                out.append(x)

        act = self.AdaptiveAvgPool3d(self.classifier(out[-1])).view(x.size(0), -1)
        return out, act


if __name__ == '__main__':
    model = X3D_xs()
    model.eval()
    model = model.cuda()
    x = torch.rand(1, 3, 9, 640, 640).cuda()

    out = model(x)
    print(out[0].shape, out[1].shape, out[2].shape)