import torch.nn as nn
from src.core import register
import torch
from torchvision.models.video import r3d_18
import math
import time
from tqdm import trange


class X3D_xs(torch.nn.Module):
    def __init__(self, feat_strides=[4, 8, 16], model_name='x3d_xs', cdim=[96, 256, 1024]):
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

class I3D18(torch.nn.Module): # 0.935, 0.5265, FPS: 133
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
    model = model.cuda().half()
    x = torch.rand(1, 3, 5, 576, 1024).cuda().half()

    t = 0
    with torch.no_grad():
        for i in trange(1000):
            start = time.time()
            action, activity = model(x)
            box, scope = action[0][:, :4], action[0][:, 4:]
            scope = torch.argmax(scope, dim=-1)
            torch.cuda.synchronize()
            t += time.time() - start
        print('FPS:', 1 / (t / 1000))