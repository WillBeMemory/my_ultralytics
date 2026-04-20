import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== 确定性全局平均池化（顶层类） ====================
class DeterministicGlobalAvgPool(nn.Module):
    def forward(self, x):
        # 使用 torch.mean，在 CUDA 上具有确定性
        return x.mean(dim=(2, 3), keepdim=True)


# ==================== 确定性坐标注意力（顶层类） ====================
class DeterministicCoordAtt(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, channels, 1)
        self.conv_w = nn.Conv2d(mip, channels, 1)

    def forward(self, x):
        identity = x
        B, C, H, W = x.shape

        # 水平方向池化 (H,1) —— 确定性
        x_h = x.mean(dim=3, keepdim=True)                 # (B, C, H, 1)
        # 垂直方向池化 (1,W)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # (B, C, W, 1)

        # 拼接并变换
        y = torch.cat([x_h, x_w], dim=2)                  # (B, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 拆分为两个方向
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)                    # (B, C, 1, W)

        # 生成注意力权重
        a_h = self.conv_h(x_h).sigmoid()                  # (B, C, H, 1)
        a_w = self.conv_w(x_w).sigmoid()                  # (B, C, 1, W)

        # 应用权重
        out = identity * a_w * a_h
        return out


# ==================== 渐进式通道-坐标注意力（确定性版本） ====================
class PCCA(nn.Module):
    def __init__(self, channels, reduction=16, stages=2):
        super().__init__()
        self.channels = channels
        self.stages = stages

        self.ca_modules = nn.ModuleList()
        self.coord_modules = nn.ModuleList()

        for _ in range(stages):
            # 通道注意力分支（确定性）
            ca = nn.Sequential(
                DeterministicGlobalAvgPool(),
                nn.Conv2d(channels, channels // reduction, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // reduction, channels, 1, bias=False),
                nn.Sigmoid()
            )
            self.ca_modules.append(ca)

            # 坐标注意力分支（确定性）
            self.coord_modules.append(DeterministicCoordAtt(channels, reduction))

        # 可学习的融合权重
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        out = x
        for i in range(self.stages):
            ca_out = self.ca_modules[i](out) * out
            coord_out = self.coord_modules[i](ca_out)
            out = self.alpha * ca_out + (1 - self.alpha) * coord_out
        return out


# ==================== 测试代码 ====================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PCCA(64, reduction=16, stages=2).to(device)
    x = torch.randn(2, 64, 32, 32).to(device)
    out = model(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    loss = out.mean()
    loss.backward()
    print("Backward pass completed (deterministic).")