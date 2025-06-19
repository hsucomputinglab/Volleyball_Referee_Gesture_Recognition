import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from src.core import register
from src.nn.modules.box_block import *  
import matplotlib.pyplot as plt
import cv2
import time

@register
class RTDETR(nn.Module):
    def __init__(self, backbone, encoder, decoder, k=15, threshold=0.5, num_classes=10, aux=[1984, 1024, 960], multi_scale=None):
        super().__init__()
        self.backbone = backbone # k100
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        self.aux = nn.ModuleList([out_classifier(aux[i], 512, 8) for i in range(len(aux))])
        self.boxes_embedding = boxes_embedding(in_channels=num_classes+4, hidden_dim=64, nhead=4, num_layers=3)
        self.train_postprocessor = train_postprocessor(k=k, threshold=threshold, num_classes=num_classes)


    def forward(self, x, targets=None):
        # img = np.ascontiguousarray(x[0][:, 2, :, :].clone().detach().permute(1, 2, 0).cpu().numpy())

        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            out = []
            for i in range(x.shape[2]):
                re = F.interpolate(x[:, :, i, :, :], size=[sz, sz]).unsqueeze(2)
                out.append(re)
            x = torch.cat(out, dim=2)

        x, act = self.backbone(x)
        # for i, feat in enumerate(x):
        #      print(f"[DEBUG] feature {i}: shape = {feat.shape}")
        x = [x[i].view(x[i].size(0), -1, x[i].size(3), x[i].size(4)) for i in range(len(x))] 
        
        x = self.encoder(x)
        x = self.decoder(x, targets)
        

        player = [self.train_postprocessor(x['pred_logits'].detach(), x['pred_boxes'].detach())]
        x['player'] = player
        player = player + [player[0][torch.randperm(player[0].size(0))] for i in range(100) if self.training]
        player = self.boxes_embedding(player)


        activity = [torch.cat([act, player[0]], dim=1).clone().detach()]
        activity.extend([act, player[0].flatten(1)]) if self.training else None
        activity = [self.aux[i](activity[i]) for i in range(len(activity))]

        x['activity'] = activity
        x['cos_boxes'] = player
        x['output'] = player[0]

        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False








        # action = player[0][:, :, 4:].cpu().numpy()
        # action = np.argmax(action, axis=2)
        # boxes = player[0][:, :, :4].cpu()
        # boxes[:, :, 0] = boxes[:, :, 0] * 1024
        # boxes[:, :, 1] = boxes[:, :, 1] * 576
        # boxes[:, :, 2] = boxes[:, :, 2] * 1024
        # boxes[:, :, 3] = boxes[:, :, 3] * 576

        # boxes[:, :, :2], boxes[:, :, 2:] = boxes[:, :, :2] - boxes[:, :, 2:] / 2, boxes[:, :, :2] + boxes[:, :, 2:] / 2
        # boxes = boxes[0].cpu().numpy()
        # for idx, box in enumerate(boxes):
        #     cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        #     cv2.putText(img, f'{action[0][0]}', (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # plt.imshow(img)
        # plt.show()


