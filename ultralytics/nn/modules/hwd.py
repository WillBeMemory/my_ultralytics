import torch
import torch.nn as nn
import torch.nn.functional as F


class HWD(nn.Module):
    """
    小波下采样模块 (HWD) —— 固定 Haar 滤波器版本。
    1. 固定 Haar 分解 → (B, 4*C_in, H/2, W/2)
    2. 1×1 卷积压缩 → (B, C_out, H/2, W/2)
    """
    def __init__(self, c1: int, c2: int, act: bool = True):
        super().__init__()
        self.c1 = c1
        self.c2 = c2

        # ---------- 固定的 Haar 分解滤波器 ----------
        w_ll = torch.tensor([[1., 1.],
                             [1., 1.]]) / 2.0
        w_lh = torch.tensor([[1., -1.],
                             [1., -1.]]) / 2.0
        w_hl = torch.tensor([[1., 1.],
                             [-1., -1.]]) / 2.0
        w_hh = torch.tensor([[1., -1.],
                             [-1., 1.]]) / 2.0
        haar_init = torch.stack([w_ll, w_lh, w_hl, w_hh], dim=0).unsqueeze(1)  # (4, 1, 2, 2)
        # 不再使用 nn.Parameter，而是注册为不可训练的 buffer
        self.register_buffer('haar_filter', haar_init.repeat(c1, 1, 1, 1))

        # ---------- 1×1 压缩卷积 ----------
        self.conv = nn.Conv2d(c1 * 4, c2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 固定 Haar 滤波器，分组卷积下采样
        x = F.conv2d(x, self.haar_filter, stride=2, groups=self.c1)
        # 1x1 压缩 + BN + 激活
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


# ============================================
# 测试代码（保持原有测试）
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