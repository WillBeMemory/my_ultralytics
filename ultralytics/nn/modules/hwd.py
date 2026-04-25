# ============================================
# File: ultralytics/nn/modules/HWD.py
# 功能：小波无损下采样模块 (HWD)
# 使用：在 backbone 中替换 stride=2 的 Conv 层
# ============================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDownsampling(nn.Module):
    """Haar 小波分解下采样（无参数）"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        # 四个子带滤波器，每个形状 (2, 2)
        w_ll = torch.tensor([[1., 1.],
                             [1., 1.]]) / 2.0
        w_lh = torch.tensor([[1., -1.],
                             [1., -1.]]) / 2.0
        w_hl = torch.tensor([[1., 1.],
                             [-1., -1.]]) / 2.0
        w_hh = torch.tensor([[1., -1.],
                             [-1., 1.]]) / 2.0

        # 堆叠并添加通道维度 → (4, 1, 2, 2)
        filters = torch.stack([w_ll, w_lh, w_hl, w_hh], dim=0).unsqueeze(1)

        # 为每个输入通道复制 → (4 * in_channels, 1, 2, 2)
        self.register_buffer('haar_filter', filters.repeat(in_channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # groups=in_channels: 每个输入通道单独滤波，输出 4 个通道
        return F.conv2d(x, self.haar_filter, stride=2, groups=self.in_channels)


class HWD(nn.Module):
    """小波下采样模块 (Haar + 1x1 压缩)"""
    def __init__(self, c1: int, c2: int, act: bool = True):
        super().__init__()
        self.haar_down = HaarDownsampling(c1)
        self.conv = nn.Conv2d(c1 * 4, c2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.haar_down(x)      # (B, 4*C, H/2, W/2)
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)

# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 模拟 YOLO11n P4 特征 (C=128, 40x40)
    x = torch.randn(2, 128, 40, 40).to(device)

    # 标准下采样对比
    conv_standard = nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False).to(device)
    hwd = HWD(128, 256).to(device)

    out_conv = conv_standard(x)
    out_hwd = hwd(x)

    print("=== 输出形状对比 ===")
    print(f"标准 Conv: {out_conv.shape}")  # [2, 256, 20, 20]
    print(f"HWD      : {out_hwd.shape}")   # [2, 256, 20, 20]

    # 参数量对比
    def count_params(module):
        return sum(p.numel() for p in module.parameters())

    print("\n=== 参数量对比 ===")
    print(f"标准 Conv: {count_params(conv_standard):,}")
    print(f"HWD      : {count_params(hwd):,}")
    print(f"HWD 参数量仅为标准卷积的 {count_params(hwd)/count_params(conv_standard)*100:.1f}%")