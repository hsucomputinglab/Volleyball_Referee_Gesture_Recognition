import torch
import torch.nn as nn
from timm.models import create_model
from src.core import register

@register
class SwinTransformerLite(nn.Module):
    def __init__(self, pretrained=True, return_idx=[1, 2, 3], img_size=(576, 1024), temporal_kernel=3, **kwargs):
        super().__init__()

        self.return_idx = return_idx
        self.temporal_kernel = temporal_kernel

        # base Swin (2D)
        self.model = create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            features_only=True,
            out_indices=return_idx,
            img_size=img_size
        )

        # Temporal conv (1D over T)
        self.temporal_conv = nn.ModuleList([
            nn.Conv1d(c, c, kernel_size=temporal_kernel, padding=temporal_kernel//2, groups=c)
            for c in [192, 384, 768]
        ])

        self.proj_layers = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            for in_ch, out_ch in zip(self.model.feature_info.channels(), [192, 384, 768])
        ])

        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        x: [B, C, T, H, W]
        """
        print(f"[VideoSwin+Temporal] input shape: {x.shape}")

        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
        x = x.view(B * T, C, H, W)

        features = self.model(x)  # list of [B*T, H, W, C]
        features = [f.permute(0, 3, 1, 2).contiguous() for f in features]  # to [B*T, C, H, W]
        features = [proj(f) for proj, f in zip(self.proj_layers, features)]

        # reshape to [B, C, T, H, W] for temporal conv
        features = [f.view(B, T, f.size(1), f.size(2), f.size(3)).permute(0, 2, 1, 3, 4).contiguous() for f in features]

        # 對每一層做 temporal conv（1D 在 T 維上）: [B, C, T, H, W] -> [B, C, T, H, W]
        features = [
            self.temporal_conv[i](f.flatten(3).mean(-1))[:, :, :, None, None].expand_as(f) if f.size(2) >= self.temporal_kernel else f
            for i, f in enumerate(features)
        ]

        # 回到 [B, T, C, H, W]，讓 RT-DETR 自行處理時間維度
        out = [f.permute(0, 2, 1, 3, 4).contiguous() for f in features]  # [B, C, T, H, W] -> [B, T, C, H, W]

        # act：每層展平成 [B*T, C, H, W] 後做池化並平均
        pooled = [
            self.AdaptiveAvgPool2d(f.view(B * T, f.size(2), f.size(3), f.size(4)))
            .view(B, T, -1).mean(1) for f in out
        ]
        act = torch.cat(pooled, dim=1)

        return out, act
