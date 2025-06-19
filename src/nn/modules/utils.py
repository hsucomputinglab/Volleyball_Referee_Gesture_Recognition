"""by lyuwenyu
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import time


def postprocessor(logits, boxes):
    logits = F.sigmoid(logits)
    scores, index = torch.topk(logits.flatten(1), 300, axis=-1)
    p = torch.sum(scores > 0.5, dim=1)
    print(p)
    # p[p > 15] = 15
    #
    #
    # index = index // 10
    # labels = index % 10
    # boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
    # logits = logits.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, logits.shape[-1]))
    # labels = F.one_hot(labels, 10)
    #
    #
    # bbox, logs, labs = [], [], []
    # for i in range(len(p)):
    #     start = time.time()
    #     b = boxes[i][:p[i]]
    #     log = logits[i][:p[i]]
    #     lab = labels[i][:p[i]]
    #
    #     x, idx = torch.topk(b[:, 0], p[i], axis=-1)
    #     b = b.gather(dim=0, index=idx.unsqueeze(-1).repeat(1, b.shape[-1]))
    #     log = log.gather(dim=0, index=idx.unsqueeze(-1).repeat(1, log.shape[-1]))
    #     lab = lab.gather(dim=0, index=idx.unsqueeze(-1).repeat(1, lab.shape[-1]))
    #
    #     b = torch.cat([b, torch.zeros((15 - p[i], b.shape[-1])).to(b.device)], dim=0)
    #     log = torch.cat([log, torch.zeros((15 - p[i], log.shape[-1])).to(log.device)], dim=0)
    #     lab = torch.cat([lab, torch.zeros((15 - p[i], lab.shape[-1])).to(lab.device)], dim=0)
    #
    #     bbox.append(b)
    #     logs.append(log)
    #     labs.append(lab)
    #
    #
    # boxes = torch.stack(bbox, dim=0)
    # logits = torch.stack(logs, dim=0)
    # labels = torch.stack(labs, dim=0)
    # player = torch.cat([boxes, labels], dim=-1)

    return player


def bbox_show(img, boxes):
    p = boxes.clone().detach().cpu().numpy() * 640
    p[:, 0:1] = p[:, 0:1] - p[:, 2:3] / 2
    p[:, 1:2] = p[:, 1:2] - p[:, 3:4] / 2
    p[:, 2:3] = p[:, 0:1] + p[:, 2:3]
    p[:, 3:4] = p[:, 1:2] + p[:, 3:4]
    p = p.astype(int)
    [cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 1) for b in p]
    plt.imshow(img)
    plt.show()


def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init

def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clip(min=0., max=1.)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[level].flatten(2).permute(
            0, 2, 1).reshape(bs * n_head, c, h, w)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].permute(
            0, 2, 1, 3, 4).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_head, 1, Len_q, n_levels * n_points)
    output = (torch.stack(
        sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).reshape(bs, n_head * c, Len_q)

    return output.permute(0, 2, 1)



