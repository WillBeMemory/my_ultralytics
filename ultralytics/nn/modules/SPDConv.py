import torch
import torch.nn as nn


class SPDConv(nn.Module):
    """
    Space-to-Depth Convolution (SPD-Conv)
    通过空间到深度重排实现无损下采样，再用 1×1 或 3×3 卷积压缩通道。

    参数:
        c1           : 输入通道数
        c2           : 输出通道数
        scale        : 下采样比例，通常为 2（尺寸减半）
        pass_through : 是否在卷积后加入残差连接（仅当 c1*scale^2 == c2 时有效）
    """
    def __init__(self, c1, c2, scale=2, kernel_size=1, pass_through=False):
        super().__init__()
        self.scale = scale
        # 核心操作：空间到深度
        self.space_to_depth = nn.PixelUnshuffle(downscale_factor=scale)
        # 随后的卷积：处理膨胀后的通道
        expanded_ch = c1 * scale ** 2
        self.conv = nn.Sequential(
            nn.Conv2d(expanded_ch, c2, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )
        self.pass_through = pass_through and expanded_ch == c2

    def forward(self, x):
        # 1. 空间到深度：尺寸减半，通道 ×4
        out = self.space_to_depth(x)   # (B, C*4, H/2, W/2)
        # 2. 卷积精炼
        out = self.conv(out)
        return out


# ====================== 测试 ======================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SPDConv(128, 256, scale=2).to(device)
    x = torch.randn(2, 128, 40, 40).to(device)
    y = model(x)
    print(f"输入: {x.shape} -> 输出: {y.shape}")   # [2, 256, 20, 20]
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")