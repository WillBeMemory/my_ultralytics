import torch
import torch.nn as nn


class SPDConv(nn.Module):
    """
    轻量无损下采样模块：空间到深度 + 深度卷积 + 1x1卷积

    无损下采样（SPD）保留所有空间细节；
    深度卷积仅在空间交互，提取每通道的局部特征；
    1x1卷积仅在通道交互，融合信息并完成降维。

    Args:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        kernel_size (int): 深度卷积的卷积核大小，默认3
        activation (nn.Module): 激活函数，默认SiLU(inplace=True)
    """

    def __init__(self, c1, c2, kernel_size=3, activation=nn.SiLU):
        super().__init__()
        # 1. 空间到深度层（无参数），将2x2空间块堆叠到通道维度
        #    输入 (B, c1, H, W) -> 输出 (B, 4*c1, H/2, W/2)
        self.spd = nn.PixelUnshuffle(downscale_factor=2)

        # 2. 深度卷积层：groups = 输入通道数，每个通道独立进行空间滤波
        #    不改变通道数，仅做空间交互
        self.depthwise = nn.Conv2d(
            c1 * 4, c1 * 4,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=c1 * 4,  # 深度卷积的关键：组数等于输入通道数
            bias=False
        )
        self.dw_bn = nn.BatchNorm2d(c1 * 4)
        self.dw_act = activation(inplace=True) if activation == nn.SiLU else activation()

        # 3. 1x1卷积层：仅在通道间交互，同时将通道数从4*c1压缩到c2
        self.pointwise = nn.Conv2d(c1 * 4, c2, kernel_size=1, stride=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(c2)
        self.pw_act = activation(inplace=True) if activation == nn.SiLU else activation()

    def forward(self, x):
        x = self.spd(x)  # (B, c1, H, W) -> (B, 4*c1, H/2, W/2)
        x = self.dw_act(self.dw_bn(self.depthwise(x)))  # 空间交互
        x = self.pw_act(self.pw_bn(self.pointwise(x)))  # 通道交互 + 降维
        return x


# ================== 简单测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 模拟输入：batch=2, 通道=64, 空间=80x80
    x = torch.randn(2, 64, 80, 80).to(device)

    # 创建模块：输入64通道，输出128通道，深度卷积核大小3
    spd_conv = SPDConv(c1=64, c2=128, kernel_size=3).to(device)
    print(spd_conv)

    # 前向传播
    y = spd_conv(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")

    # 验证输出尺寸：应为 (2, 128, 40, 40)
    assert y.shape == (2, 128, 40, 40), f"Shape mismatch! Expected (2, 128, 40, 40), got {y.shape}"

    # 计算参数量和粗略计算量
    total_params = sum(p.numel() for p in spd_conv.parameters())
    # 计算量（MACs）估算（仅乘加）
    # 深度卷积 MACs = 4*c1 * kernel_size^2 * H/2 * W/2
    # 点卷积 MACs   = c2 * (4*c1) * H/2 * W/2
    from math import prod

    H_out, W_out = 40, 40
    macs_dw = (64 * 4) * (3 ** 2) * H_out * W_out
    macs_pw = 128 * (64 * 4) * H_out * W_out
    total_macs = macs_dw + macs_pw
    print(f"Total parameters: {total_params:,}")
    print(f"Estimated MACs: {total_macs:,} (≈ {total_macs / 1e6:.2f} M)")

    print("Test passed!")