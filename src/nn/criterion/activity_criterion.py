from src.core import register
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# from torchvision.ops import box_convert, generalized_box_iou
from src.nn.modules.box_ops import box_cxcywh_to_xyxy, box_iou
from src.misc.dist import get_world_size, is_dist_available_and_initialized
import torch.nn as nn
import time


class ActivityCriterion(nn.Module):
    def __init__(self):
        super(ActivityCriterion, self).__init__()
        self.binary = Binary()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.cosine_embedding = nn.CosineEmbeddingLoss()

    def forward(self, outputs, target):

        loss_dict = {}
        activity = outputs['activity']
        player = outputs['cos_boxes']

        for i in range(len(activity)):
            loss_dict['activity_' + str(i)] = self.cross_entropy(activity[i], target) + self.binary(activity[i], target)

        ones = torch.ones((player[i].size(0))).to(player[i].device)
        cosine = sum([self.cosine_embedding(player[i], player[i + 1], ones) for i in range(len(player) - 1)]) / (len(player) - 1)
        l1 = sum([F.l1_loss(player[0], player[i + 1]) for i in range(len(player) - 1)]) / (len(player) - 1)
        loss_dict['cosine'] = cosine + l1

        return loss_dict


class Binary(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(Binary, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        target = F.one_hot(target.long(), num_classes=8).to(predict.device)
        predict = torch.sigmoid(predict)

        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.sub(predict, target).pow(self.p), dim=1)
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1)  # + self.smooth

        loss = num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


# class ActivityCriterion(nn.Module):
#     def __init__(self):
#         super(ActivityCriterion, self).__init__()
#         self.binary = Binary()
#         self.cross_entropy = nn.CrossEntropyLoss()
#         self.cosine_embedding = nn.CosineEmbeddingLoss()
#
#     def forward(self, outputs, target):
#
#         loss_dict = {}
#         activity = outputs['activity']
#         player = outputs['cos_boxes']
#
#         for i in range(len(activity)):
#             loss_dict['activity_' + str(i)] = self.cross_entropy(activity[i], target) + self.binary(activity[i], target)
#
#         ones = torch.ones((player[i].size(0))).to(player[i].device)
#         cosine = sum([self.cosine_embedding(player[i], player[i + 1], ones) for i in range(len(player) - 1)]) / (len(player) - 1)
#         loss_dict['cosine'] = cosine
#
#         return loss_dict