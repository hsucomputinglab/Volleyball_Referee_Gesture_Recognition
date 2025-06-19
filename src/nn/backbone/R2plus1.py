import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18
from src.core import register

@register
class R2Plus1D18(nn.Module):
    def __init__(self, pretrained=True, num_level=3, cdim=[128, 256, 512]):
        super().__init__()
        self.model = r2plus1d_18(pretrained=pretrained)
        self.num_level = num_level
        self.stem = self.model.stem  # Conv3D + BN + ReLU + MaxPool
        self.blocks = nn.ModuleList([
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
        ])
        self.AdaptiveAvgPool3d = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        """
        x: [B, C, T, H, W]
        return: list of [B, C, H, W], act: [B, D]
        """
        B, C, T, H, W = x.shape
        out = []
        x = self.stem(x)
        
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i >= len(self.blocks) - self.num_level:
                # For features with temporal dimension > 1, take the last frame or average across time
                if x.size(2) > 1:
                    # Option 1: Take the last frame
                    features = x[:, :, -1, :, :]  # [B, C, H, W]
                    
                    # Option 2: Average across time dimension
                    # features = torch.mean(x, dim=2)  # [B, C, H, W]
                else:
                    features = x.squeeze(2)  # Remove temporal dimension when it's 1
                    
                out.append(features)
        
        # Create activation features for auxiliary tasks
        act = torch.cat([
            self.AdaptiveAvgPool3d(f.unsqueeze(2) if f.dim() == 4 else f).view(f.size(0), -1)
            for i, f in enumerate(out)
        ], dim=1)
        
        return out, act