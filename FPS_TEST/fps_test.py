import os
import time
import torch
import pandas as pd
from tqdm import tqdm
from torchvision.models.video import r3d_18
from torch import nn
import math


# =========================
#  I3D-18 定義
# =========================
class I3D18(nn.Module):
    def __init__(self, num_level=3):
        super(I3D18, self).__init__()
        self.model = r3d_18(pretrained=False)
        self.num_level = num_level
        self.stem = self.model.stem
        self.blocks = nn.ModuleList([
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
        ])
        self.AdaptiveAvgPool3d = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        out = []
        x = self.stem(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i >= len(self.blocks) - self.num_level:
                out.append(x)
        act = torch.cat([self.AdaptiveAvgPool3d(o).view(x.size(0), -1) for o in out], 1)
        return out, act


# =========================
#  X3D-xs 定義
# =========================
class X3D_xs(nn.Module):
    def __init__(self, feat_strides=[4, 8, 16], model_name='x3d_xs', cdim=[96, 256, 1024]):
        super(X3D_xs, self).__init__()
        self.model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
        self.classifier = self.model.blocks[-1].pool
        self.classifier.pre_conv = nn.Conv3d(cdim[0], cdim[1], kernel_size=(1, 1, 1), bias=False)
        self.classifier.pre_norm = nn.BatchNorm3d(cdim[1])
        self.classifier.post_conv = nn.Conv3d(cdim[1], cdim[2], kernel_size=(1, 1, 1), bias=False)
        self.AdaptiveAvgPool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.model = self.model.blocks[:int(math.log(feat_strides[-1], 2))]
        self.num_levels = len(feat_strides)

    def forward(self, x):
        out = []
        for i in range(len(self.model)):
            x = self.model[i](x)
            if i >= len(self.model) - self.num_levels:
                out.append(x)
        act = self.AdaptiveAvgPool3d(self.classifier(out[-1])).view(x.size(0), -1)
        return out, act


# =========================
#  設定與測試流程
# =========================
def test_fps(model_type="i3d", weights_dir="./weights", output_csv="fps_results.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.rand(1, 3, 5, 576, 1024).to(device).half()
    all_weights = [f for f in os.listdir(weights_dir) if f.endswith(".pth")]
    results = []

    for weight_name in tqdm(all_weights):
        # 選擇模型架構
        if model_type == "i3d":
            model = I3D18().to(device).half()
        elif model_type == "x3d":
            model = X3D_xs().to(device).half()
        else:
            raise ValueError("Unsupported model type. Use 'i3d' or 'x3d'.")

        model.eval()
        weight_path = os.path.join(weights_dir, weight_name)

        try:
            state_dict = torch.load(weight_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)

            total_time = 0
            with torch.no_grad():
                for _ in range(5000):
                    start = time.time()
                    out, act = model(input_tensor)
                    torch.cuda.synchronize()
                    total_time += time.time() - start

            fps = 1 / (total_time / 5000)
            results.append({'weight': weight_name, 'fps': round(fps, 2)})
        except Exception as e:
            results.append({'weight': weight_name, 'fps': 'Error'})
            print(f"[Error] {weight_name}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ 測試完成，結果已儲存至 {output_csv}")


# =========================
#  執行
# =========================
if __name__ == "__main__":
    # 可選 "i3d" 或 "x3d"
    test_fps(model_type="i3d", weights_dir="./weights_i3d", output_csv="fps_i3d_5000.csv")
    test_fps(model_type="x3d", weights_dir="./weights_x3d", output_csv="fps_x3d_5000.csv")
