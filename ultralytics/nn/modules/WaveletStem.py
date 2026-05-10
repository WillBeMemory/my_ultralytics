import torch
import torch.nn as nn
import torch.nn.functional as F


# ================== 修正后的 Haar 小波滤波器生成 ==================
def haar_filters(in_channels: int):
    """
    生成形状为 (in_channels*4, 1, 2, 2) 的滤波器，
    保证在 groups=in_channels 下每个通道组内顺序为 [LL, LH, HL, HH]。
    """
    dec_lo = torch.tensor([1 / 2, 1 / 2])
    dec_hi = torch.tensor([1 / 2, -1 / 2])
    # 四个基滤波器，形状 (1,1,2,2)
    base_LL = (dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    base_LH = (dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    base_HL = (dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    base_HH = (dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    # 拼接成 (1,4,2,2)，即一个通道的四个滤波器
    per_channel = torch.cat([base_LL, base_LH, base_HL, base_HH], dim=1)  # (1,4,2,2)
    # 重复 in_channels 次 → (in_channels, 4, 2, 2)，然后 reshape 为 (4*in_channels, 1, 2, 2)
    per_channel = per_channel.repeat(in_channels, 1, 1, 1)
    return per_channel.reshape(in_channels * 4, 1, 2, 2)


def wavelet_decompose(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    return F.conv2d(x, filters, stride=2, groups=C, padding=0)


# ================== 方向性子带卷积 ==================
class DirectionalConv(nn.Module):
    """
    根据不同子带特性设计的方向卷积：
    - LL: 标准 3×3 深度卷积
    - LH: (1,3) 垂直条状深度卷积（增强水平边缘）
    - HL: (3,1) 水平条状深度卷积（增强垂直边缘）
    - HH: 3×3 深度卷积，可初始化为对角模式（强化对角边缘）
    """
    def __init__(self, channels: int, mode: str, use_diag_init: bool = True):
        super().__init__()
        self.mode = mode
        if mode == 'll':
            self.conv = nn.Conv2d(channels, channels, 3, padding=1,
                                  groups=channels, bias=False)
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
        """初始化 HH 卷积核为对角增强模式（主对角线正，反对角线负）"""
        with torch.no_grad():
            weight = self.conv.weight  # (C, 1, 3, 3)
            nn.init.zeros_(weight)
            # 主对角线 (+)
            weight[:, 0, 0, 0] = 1.0
            weight[:, 0, 1, 1] = 1.0
            weight[:, 0, 2, 2] = 1.0
            # 反对角线 (-)
            weight[:, 0, 0, 2] = -0.5
            weight[:, 0, 1, 1] -= 1.0  # 中心点被正对角线占用，适当调整
            weight[:, 0, 2, 0] = -0.5
            # 归一化
            C = weight.shape[0]
            weight /= weight.view(C, -1).norm(p=2, dim=1).view(C, 1, 1, 1) + 1e-6

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ================== 修正后的双分支 WaveletStem ==================
class WaveletStem(nn.Module):
    """
    双分支茎模块（小波分支使用正确的滤波器分解）：
    分支1: 小波分解 → 方向卷积 → 拼接扩张
    分支2: 标准 3×3 stride2 卷积
    两分支拼接后 1×1 融合输出。
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 64,
                 use_diag_init: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ----- 小波分支 -----
        self.register_buffer('dec_filters', haar_filters(in_channels))
        self.ll_conv = DirectionalConv(in_channels, 'll')
        self.lh_conv = DirectionalConv(in_channels, 'lh')
        self.hl_conv = DirectionalConv(in_channels, 'hl')
        self.hh_conv = DirectionalConv(in_channels, 'hh', use_diag_init=use_diag_init)
        self.wavelet_expand = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

        # ----- 标准卷积分支 (stride=2 下采样) -----
        self.conv_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

        # ----- 融合层 (2*out_channels -> out_channels) -----
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # 填充到偶数尺寸（Haar 分解要求）
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # 分支1: 小波路径
        coeffs = wavelet_decompose(x, self.dec_filters)      # (B, 4*C, H/2, W/2)
        B, C4, H2, W2 = coeffs.shape
        C_orig = C4 // 4
        coeffs = coeffs.view(B, C_orig, 4, H2, W2)
        ll = coeffs[:, :, 0, :, :]
        lh = coeffs[:, :, 1, :, :]
        hl = coeffs[:, :, 2, :, :]
        hh = coeffs[:, :, 3, :, :]

        # 方向卷积增强
        ll = self.ll_conv(ll)
        lh = self.lh_conv(lh)
        hl = self.hl_conv(hl)
        hh = self.hh_conv(hh)

        wavelet_feat = torch.cat([ll, lh, hl, hh], dim=1)    # (B, 4*C_orig, H/2, W/2)
        wavelet_feat = self.wavelet_expand(wavelet_feat)      # (B, out_channels, H/2, W/2)

        # 分支2: 标准卷积分支
        conv_feat = self.conv_branch(x)                       # (B, out_channels, H/2, W/2)

        # 融合
        combined = torch.cat([wavelet_feat, conv_feat], dim=1)
        out = self.fusion(combined)
        return out


# ================== 测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 偶数尺寸
    x_even = torch.randn(2, 3, 640, 640).to(device)
    # 奇数尺寸
    x_odd = torch.randn(2, 3, 641, 641).to(device)

    model = WaveletStem(in_channels=3, out_channels=64, use_diag_init=True).to(device)
    model.eval()

    with torch.no_grad():
        out_even = model(x_even)
        out_odd = model(x_odd)

    print(f"Even input: {x_even.shape} -> output: {out_even.shape}")
    print(f"Odd input: {x_odd.shape} -> output: {out_odd.shape}")
    assert out_even.shape == (2, 64, 320, 320), f"Shape mismatch: {out_even.shape}"
    print("✅ Output shapes verified.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")