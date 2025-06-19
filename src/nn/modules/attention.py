import math
from torch import nn
import torch
import time


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        batch_size, head, length, d_tensor = k.size()
        k_t = k.transpose(2, 3)
        attn = (q @ k_t) / math.sqrt(d_tensor)
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float("-inf"))

        attn = self.dropout(self.softmax(attn))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaledDotProductAttention(dropout)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_concat = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, attn_mask=None):
        bs = q.size(0)
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).repeat(bs, 1, 1)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        score, attn = self.attention(q, k, v, attn_mask=attn_mask)
        score = self.concat(score)
        score = self.w_concat(score)
        return score, attn

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        return tensor

    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


if __name__ == '__main__':
    model = MultiHeadAttention(256, 8)
    model.eval()
    model = model.cuda()
    mask = torch.zeros(300, 300).cuda()
    x = torch.rand(5, 300, 256).cuda()

    out = model(x, x, x, mask)
    print(out[0].shape)