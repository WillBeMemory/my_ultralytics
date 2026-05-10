import torch
import torch.nn as nn
import torch.nn.functional as F


# ================== 正确的 Haar 滤波器生成 ==================
def haar_filters_groups(in_channels: int):
    dec_lo = torch.tensor([1 / 2, 1 / 2])
    dec_hi = torch.tensor([1 / 2, -1 / 2])
    base_LL = (dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    base_LH = (dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    base_HL = (dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    base_HH = (dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    per_channel = torch.cat([base_LL, base_LH, base_HL, base_HH], dim=1)  # (1,4,2,2)
    per_channel = per_channel.repeat(in_channels, 1, 1, 1)  # (C,4,2,2)
    return per_channel.reshape(in_channels * 4, 1, 2, 2)


def wt_decompose_level(x: torch.Tensor):
    B, C, H, W = x.shape
    filters = haar_filters_groups(C).to(device=x.device, dtype=x.dtype)
    coeffs = F.conv2d(x, filters, stride=2, groups=C, padding=0)
    coeffs = coeffs.view(B, C, 4, H // 2, W // 2)
    return coeffs[:, :, 0], coeffs[:, :, 1], coeffs[:, :, 2], coeffs[:, :, 3]


# ================== 方向卷积（可支持标准卷积作为 LL） ==================
class DirectionalConv(nn.Module):
    def __init__(self, channels: int, mode: str, use_diag_init: bool = True):
        super().__init__()
        if mode == 'll':
            # 标准 3×3 卷积（groups=1，不使用深度卷积）
            self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        elif mode == 'lh':
            self.conv = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1),
                                  groups=channels, bias=False)
        elif mode == 'hl':
            self.conv = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0),
                                  groups=channels, bias=False)
        elif mode == 'hh':
            self.conv = nn.Conv2d(channels, channels, 3, padding=1,
                                  groups=channels, bias=False)
            if use_diag_init:
                self._init_diag_weights()
        else:
            raise ValueError(f"Invalid mode: {mode}")
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def _init_diag_weights(self):
        with torch.no_grad():
            weight = self.conv.weight  # (C, 1, 3, 3)
            nn.init.zeros_(weight)
            weight[:, 0, 0, 0] = 1.0
            weight[:, 0, 1, 1] = 1.0
            weight[:, 0, 2, 2] = 1.0
            weight[:, 0, 0, 2] = -0.5
            weight[:, 0, 1, 1] -= 1.0
            weight[:, 0, 2, 0] = -0.5
            C = weight.shape[0]
            weight /= weight.view(C, -1).norm(p=2, dim=1).view(C, 1, 1, 1) + 1e-6

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ================== HWD：LL 标准卷积 + LH/HL/HH 方向卷积 ==================
class HWD(nn.Module):
    """
    结构化小波下采样模块（LL 标准卷积，高频方向卷积）。

    流程：
    1. Haar 小波分解 → LL, LH, HL, HH
    2. 各子带增强：
       - LL: 标准 3×3 卷积 + BN + SiLU
       - LH: (1,3) 深度卷积 + BN + SiLU （增强水平边缘）
       - HL: (3,1) 深度卷积 + BN + SiLU （增强垂直边缘）
       - HH: 3×3 深度卷积 + BN + SiLU （对角增强，可初始化）
    3. 各自通过独立的 1×1 卷积压缩到 out_channels // 4
    4. 按 [LL, LH, HL, HH] 顺序拼接
    5. BatchNorm + SiLU 激活
    """

    def __init__(self, c1: int, c2: int, use_diag_init: bool = True):
        super().__init__()
        assert c2 % 4 == 0, f"c2 must be divisible by 4, got {c2}"
        self.c1 = c1
        self.c2 = c2
        ch_per_band = c2 // 4

        # 四个子带的增强卷积
        self.ll_conv = DirectionalConv(c1, 'll')  # 标准卷积
        self.lh_conv = DirectionalConv(c1, 'lh')
        self.hl_conv = DirectionalConv(c1, 'hl')
        self.hh_conv = DirectionalConv(c1, 'hh', use_diag_init=use_diag_init)

        # 压缩到目标通道数的 1/4
        self.ll_proj = nn.Conv2d(c1, ch_per_band, 1, bias=False)
        self.lh_proj = nn.Conv2d(c1, ch_per_band, 1, bias=False)
        self.hl_proj = nn.Conv2d(c1, ch_per_band, 1, bias=False)
        self.hh_proj = nn.Conv2d(c1, ch_per_band, 1, bias=False)

        # 拼接后的 BN + 激活
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # 填充到偶数尺寸
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # 正确的小波分解
        LL, LH, HL, HH = wt_decompose_level(x)  # 各 (B, C, H/2, W/2)

        # 子带增强 + 压缩
        ll = self.ll_proj(self.ll_conv(LL))  # (B, ch_per_band, H/2, W/2)
        lh = self.lh_proj(self.lh_conv(LH))
        hl = self.hl_proj(self.hl_conv(HL))
        hh = self.hh_proj(self.hh_conv(HH))

        # 按顺序拼接
        out = torch.cat([ll, lh, hl, hh], dim=1)  # (B, c2, H/2, W/2)
        return self.act(self.bn(out))


# ================== 简单测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 实例化 HWD：输入 3 通道，输出 64 通道
    hwd = HWD(c1=3, c2=64).to(device)
    x = torch.randn(2, 3, 256, 256).to(device)
    out = hwd(x)
    print(f"Input  shape: {x.shape}")
    print(f"Output shape: {out.shape}")  # 期望 (2, 64, 128, 128)
    assert out.shape == (2, 64, 128, 128), f"Shape mismatch: {out.shape}"
    print("✅ Output shape verified.")

    # 参数量
    total_params = sum(p.numel() for p in hwd.parameters())
    print(f"Total parameters: {total_params:,}")

    # 奇数尺寸测试
    x_odd = torch.randn(2, 3, 257, 257).to(device)
    out_odd = hwd(x_odd)
    print(f"Odd input shape: {x_odd.shape} -> Output shape: {out_odd.shape}")
    assert out_odd.shape[0] == 2 and out_odd.shape[1] == 64
    print("✅ Odd size handled correctly.")