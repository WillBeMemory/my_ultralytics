import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvElection(nn.Module):
    """
    基于卷积的学习选举模块。
    通过轻量卷积预测 2 组 × 4 个权重，对 SPD 后的 4 个偏移像素进行加权聚合，压缩为 2C 通道。
    """
    def __init__(self, channels, mid_ratio=0.5):
        super().__init__()
        self.channels = channels
        mid = max(1, int(channels * 4 * mid_ratio))

        # 预测网络：输出 8 通道（2 组 × 4 像素权重）
        self.pred = nn.Sequential(
            nn.Conv2d(channels * 4, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, 8, kernel_size=1, bias=False)  # 8 = 2 heads × 4 pixels
        )

    def forward(self, x):
        # x: (B, 4C, H, W)
        B, C4, H, W = x.shape
        C = self.channels

        # 预测权重 (B, 8, H, W)
        w = self.pred(x)                       # (B, 8, H, W)
        w = w.view(B, 2, 4, H, W)              # (B, 2, 4, H, W)  -> 2 个输出头，每个头有 4 个像素的权重

        # 对每个头进行 softmax，使其在 4 个像素间归一化
        w = F.softmax(w, dim=2)               # 在 dim=2 上归一化

        # 重塑特征图为 (B, C, 4, H, W)
        x_blocks = x.view(B, C, 4, H, W)      # (B, C, 4, H, W)

        # 加权聚合
        # w: (B, 2, 4, H, W) -> unsqueeze(2) -> (B, 2, 1, 4, H, W)
        # x_blocks: (B, C, 4, H, W) -> unsqueeze(1) -> (B, 1, C, 4, H, W)
        w_expand = w.unsqueeze(2)              # (B, 2, 1, 4, H, W)
        x_expand = x_blocks.unsqueeze(1)       # (B, 1, C, 4, H, W)

        # 相乘并求和 dim=3（像素维度） => (B, 2, C, H, W)
        out = (w_expand * x_expand).sum(dim=3)

        # 重塑为 (B, 2*C, H, W)
        out = out.reshape(B, 2 * C, H, W)

        # 信息完整性缩放因子（此处不进行额外缩放，返回全1以便后续模块可选择使用）
        info_scale = torch.ones(B, 1, H, W, device=out.device, dtype=out.dtype)
        return out, info_scale


class SPDConv(nn.Module):
    """
    轻量无损下采样模块（卷积学习选举 + 深度可分离卷积 + 1x1降维）

    参数:
        c1, c2      : 输入/输出通道数
        kernel_size : 深度卷积核大小
        activation  : 激活函数
        mid_ratio   : 选举模块中预测网络的中间通道比例
    """
    def __init__(self, c1, c2, kernel_size=3, activation=nn.SiLU, mid_ratio=0.5):
        super().__init__()
        self.spd = nn.PixelUnshuffle(downscale_factor=2)
        self.election = ConvElection(c1, mid_ratio=mid_ratio)

        self.depthwise = nn.Conv2d(2 * c1, 2 * c1, kernel_size, padding=kernel_size // 2,
                                   groups=2 * c1, bias=False)
        self.dw_bn = nn.BatchNorm2d(2 * c1)
        self.pointwise = nn.Conv2d(2 * c1, c2, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(c2)
        self.act = activation(inplace=True) if activation == nn.SiLU else activation()

    def forward(self, x):
        x = self.spd(x)                     # (B, 4*c1, H/2, W/2)
        x, info_scale = self.election(x)    # (B, 2*c1, H/2, W/2), info_scale (B,1,H/2,W/2)

        x = self.act(self.dw_bn(self.depthwise(x)))
        x = self.act(self.pw_bn(self.pointwise(x)))
        # info_scale 当前为全1，如需启用可取消注释下一行
        # x = x * info_scale
        return x


# ================== 简单测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    x = torch.randn(2, 16, 80, 80).to(device)
    spd_conv = SPDConv(c1=16, c2=32, kernel_size=3).to(device)
    print(spd_conv)

    y = spd_conv(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")

    assert y.shape == (2, 32, 40, 40), f"Shape mismatch! Expected (2,32,40,40), got {y.shape}"

    loss = y.sum()
    loss.backward()
    print("Backward pass succeeded.")

    total_params = sum(p.numel() for p in spd_conv.parameters())
    print(f"Total parameters: {total_params:,}")
    print("Test passed!")