import torch
import torch.nn as nn


def channel_shuffle(x, groups):
    """
    通道混洗：将通道分成 groups 组，然后转置并展平，实现跨组信息混合
    Args:
        x: (B, C, H, W) 输入特征图
        groups: 分组数
    Returns:
        (B, C, H, W) 混洗后的特征图
    """
    B, C, H, W = x.shape
    x = x.view(B, groups, C // groups, H, W)
    x = x.transpose(1, 2).contiguous()
    return x.view(B, C, H, W)


class SPDConv(nn.Module):
    """
    轻量无损下采样模块（分组标准卷积 + 通道混洗 + 1×1降维）

    空间到深度 → 分组标准卷积（组数=输入通道数c1，组内4通道全交互） → 通道混洗 → 1×1降维

    - SPD 保留所有空间细节，尺寸减半，通道数扩大 4 倍（c1 → 4*c1）
    - 分组卷积：将 4*c1 个通道分成 c1 组，每组 4 通道，组内标准 3×3 卷积（组内通道全交互）
    - 通道混洗：零参数打破组间隔离，促进跨组信息流通
    - 1×1 卷积：融合所有通道并降维到目标通道 c2

    Args:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        kernel_size (int): 分组卷积的核大小，默认 3
        activation (nn.Module): 激活函数，默认 SiLU(inplace=True)
    """
    def __init__(self, c1, c2, kernel_size=3, activation=nn.SiLU):
        super().__init__()
        # 1. 空间到深度（无损下采样），输出 (4*c1, H/2, W/2)
        self.spd = nn.PixelUnshuffle(downscale_factor=2)

        # 2. 分组标准卷积：groups = c1，每组输入/输出均为 4 通道，组内标准卷积实现通道全交互
        self.group_conv = nn.Conv2d(
            4 * c1, 4 * c1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=c1,          # 分成 c1 组，每组 4 个通道
            bias=False
        )
        self.group_bn = nn.BatchNorm2d(4 * c1)
        self.group_act = activation(inplace=True) if activation == nn.SiLU else activation()

        # 3. 1×1 卷积降维（通道交互 + 压缩到 c2）
        self.pointwise = nn.Conv2d(4 * c1, c2, kernel_size=1, stride=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(c2)
        self.pw_act = activation(inplace=True) if activation == nn.SiLU else activation()

    def forward(self, x):
        # 无损下采样
        x = self.spd(x)                                          # (B, 4*c1, H/2, W/2)

        # 分组卷积（组内通道交互 + 空间滤波）
        x = self.group_act(self.group_bn(self.group_conv(x)))   # (B, 4*c1, H/2, W/2)

        # 通道混洗：打破组间壁垒，为后续 1×1 提供更好的初始化
        x = channel_shuffle(x, groups=self.group_conv.groups)   # groups = c1

        # 1×1 降维
        x = self.pw_act(self.pw_bn(self.pointwise(x)))          # (B, c2, H/2, W/2)
        return x


# ================== 简单测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 模拟输入：batch=2, 通道=16, 空间=80x80
    x = torch.randn(2, 16, 80, 80).to(device)

    # 创建模块：输入16通道，输出32通道
    spd_conv = SPDConv(c1=16, c2=32, kernel_size=3).to(device)
    print(spd_conv)

    # 前向传播
    y = spd_conv(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")

    # 验证输出尺寸：应为 (2, 32, 40, 40)
    assert y.shape == (2, 32, 40, 40), f"Shape mismatch! Expected (2, 32, 40, 40), got {y.shape}"

    # 参数量与计算量（粗略）
    total_params = sum(p.numel() for p in spd_conv.parameters())
    print(f"Total parameters: {total_params:,}")
    print("Test passed!")