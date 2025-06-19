import torch
import torch.nn as nn
import torch.nn.functional as F
from src.nn.modules.transformer import *
import time



class boxes_embedding(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, nhead=4, num_layers=3):
        super(boxes_embedding, self).__init__()
        self.box_proj = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        encoder_layer = TransformerEncoderLayer(c1=hidden_dim, num_heads=nhead)
        self.box_encoder = TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        x = [x] if not isinstance(x, list) else x
        for i in range(len(x)):
            bs, n, a = x[i].size()
            x[i] = x[i].view(bs * n, a)
            x[i] = self.box_proj(x[i]).view(bs, n, -1)
            x[i] = self.box_encoder(x[i]).flatten(1)

        return x


class train_postprocessor(nn.Module):
    def __init__(self, k=15, threshold=0.7, num_classes=10):
        super().__init__()
        self.k = k
        self.threshold = threshold
        self.num_classes = num_classes


    def forward(self, logits, boxes):
        logits = F.sigmoid(logits)
        scores, index = torch.topk(logits.flatten(1), self.k, axis=-1)

        self.zero = torch.zeros_like(scores)
        self.ones = torch.ones_like(scores)
        use = torch.where(scores < self.threshold , self.zero, scores)
        use = torch.where(use >= self.threshold , self.ones, use)
        log = use[:, :, None].repeat(1, 1, self.num_classes)
        box = use[:, :, None].repeat(1, 1, 4)

        index = index // self.num_classes
        boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1])) * box
        logits = logits.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, logits.shape[-1])) * log

        player = torch.cat([boxes, logits], dim=-1)
        indices = torch.topk(player[:, :, 0], k=self.k).indices
        player = player.gather(dim=1, index=indices.unsqueeze(-1).repeat(1, 1, player.shape[-1]))[:, :self.k, :]

        return player


class out_classifier(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, num_classes=8):
        super(out_classifier, self).__init__()
        self.out = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        x = self.out(x)
        return x
