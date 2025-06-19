import torch
import torch.nn as nn
from src.core import register
from torchvision.models.video import r3d_18
import math
import time


@register
class I3D18(torch.nn.Module): 
    def __init__(self, num_level=3, cdim=[128, 256, 512]):
        super(I3D18, self).__init__()
        self.model = r3d_18(pretrained=True)
        self.num_level = num_level
        self.stem = self.model.stem
        self.blocks = nn.ModuleList([
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
        ])

        # self.classifier = classifier(in_channels=512, out_channels=512, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.AdaptiveAvgPool3d = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = []
        x = self.stem(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i >= len(self.blocks) - self.num_level:
                out.append(x)

        # act = self.AdaptiveAvgPool3d(self.classifier(out[-1])).view(x.size(0), -1)
        act = torch.cat([self.AdaptiveAvgPool3d(out[i]).view(x.size(0), -1) for i in range(len(out))], 1)

        return out, act


if __name__ == '__main__':
    model = I3D18()
    model.eval()
    model = model.cuda()
    x = torch.rand(1, 3, 9, 640, 640).cuda()

    out, act = model(x)
    print(out[0].shape, out[1].shape, out[2].shape)
    print(act.shape)