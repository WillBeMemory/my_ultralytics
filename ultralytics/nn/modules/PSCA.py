import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialCoordinateAttention(nn.Module):
    """
    单阶段空间-坐标协同注意力模块（基础块）
    输入: (B, C, H, W)
    输出: (B, C, H, W)
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # ========== 坐标注意力分支 ==========
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, channels, 1)
        self.conv_w = nn.Conv2d(mip, channels, 1)

        # ========== 空间注意力分支 ==========
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

        # ========== 可学习融合权重 ==========
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        identity = x
        B, C, H, W = x.shape

        # ----- 坐标注意力 -----
        x_h = self.pool_h(x)                     # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0,1,3,2)    # (B, C, W, 1)

        y = torch.cat([x_h, x_w], dim=2)         # (B, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0,1,3,2)               # (B, C, 1, W)

        a_h = self.conv_h(x_h).sigmoid()          # (B, C, H, 1)
        a_w = self.conv_w(x_w).sigmoid()          # (B, C, 1, W)
        ca_out = identity * a_w * a_h

        # ----- 空间注意力（作用于坐标注意力输出）-----
        avg_out = torch.mean(ca_out, dim=1, keepdim=True)
        max_out, _ = torch.max(ca_out, dim=1, keepdim=True)
        spatial_attn = self.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        sa_out = ca_out * spatial_attn

        # ----- 可学习融合 -----
        alpha = torch.sigmoid(self.alpha)          # 限制在 (0,1)
        out = alpha * ca_out + (1 - alpha) * sa_out
        return out


class PSCA(nn.Module):
    """
    Progressive Spatial-Coordinate Attention (PSCA)
    通过堆叠多个单阶段协同注意力块，逐步精炼特征，并加入残差连接。
    参数:
        channels: 输入/输出通道数
        stages: 渐进阶段数（默认2）
        reduction: 通道压缩比（用于坐标注意力）
        kernel_size: 空间注意力卷积核大小
    """
    def __init__(self, channels, stages=2, reduction=16, kernel_size=7):
        super().__init__()
        self.stages = stages
        self.attn_blocks = nn.ModuleList([
            SpatialCoordinateAttention(channels, reduction, kernel_size)
            for _ in range(stages)
        ])
        # 可选：最终输出前的残差连接缩放因子（可学习）
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        out = x
        for block in self.attn_blocks:
            out = block(out)          # 逐阶段精炼
        # 残差连接（保留原始输入，增强梯度流）
        out = out + self.res_scale * x
        return out


# ==================== 测试代码 ====================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 测试单阶段基础块
    base_block = SpatialCoordinateAttention(64).to(device)
    x = torch.randn(2, 64, 32, 32).to(device)
    out_base = base_block(x)
    print(f"Base block - Input shape: {x.shape}, Output shape: {out_base.shape}")

    # 测试 PSCA（2阶段）
    psca = PSCA(64, stages=2).to(device)
    out_psca = psca(x)
    print(f"PSCA - Input shape: {x.shape}, Output shape: {out_psca.shape}")

    # 梯度测试
    loss = out_psca.mean()
    loss.backward()
    print("Backward pass completed.")