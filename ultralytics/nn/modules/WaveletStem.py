import torch
import torch.nn as nn
import torch.nn.functional as F


# ================== Haar 小波滤波器 ==================
def haar_filters(in_channels: int):
    dec_lo = torch.tensor([1 / 2, 1 / 2])
    dec_hi = torch.tensor([1 / 2, -1 / 2])
    base = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),  # LL
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),  # LH
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),  # HL
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)   # HH
    ], dim=0)  # (4, 2, 2)
    filters = base[:, None].repeat(1, in_channels, 1, 1)  # (4, C, 2, 2)
    return filters.reshape(in_channels * 4, 1, 2, 2)


def wavelet_decompose(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    return F.conv2d(x, filters, stride=2, groups=C, padding=0)


# ================== 子带阈值去噪 ==================
class SubbandThreshold(nn.Module):
    """简单的子带阈值去噪：LL 可学习缩放，LH/HL/HH 软阈值"""
    def __init__(self, channels: int):
        super().__init__()
        self.ll_alpha = nn.Parameter(torch.tensor(2.0))           # LL 缩放因子
        self.lh_hl_thr = nn.Parameter(torch.tensor(-4.0))          # LH/HL 共享阈值
        self.hh_thr = nn.Parameter(torch.full((channels,), -4.0))  # HH 逐通道阈值

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        # coeffs: (B, C, 4, H/2, W/2)
        ll = coeffs[:, :, 0, :, :]
        lh = coeffs[:, :, 1, :, :]
        hl = coeffs[:, :, 2, :, :]
        hh = coeffs[:, :, 3, :, :]

        ll = ll * torch.sigmoid(self.ll_alpha)
        thr_lh_hl = F.softplus(self.lh_hl_thr)
        lh = torch.sign(lh) * F.relu(torch.abs(lh) - thr_lh_hl)
        hl = torch.sign(hl) * F.relu(torch.abs(hl) - thr_lh_hl)
        thr_hh = F.softplus(self.hh_thr).view(1, -1, 1, 1)
        hh = torch.sign(hh) * F.relu(torch.abs(hh) - thr_hh)

        return torch.stack([ll, lh, hl, hh], dim=2)  # (B, C, 4, H/2, W/2)


# ================== SPDConv 模块 ==================
class SPDConv(nn.Module):
    """Space-to-Depth 卷积：pixel_unshuffle + 1x1 压缩"""
    def __init__(self, c1, c2, factor=2, compress=True):
        super().__init__()
        self.factor = factor
        self.compress = compress
        expanded_ch = c1 * factor ** 2
        if compress:
            self.conv = nn.Sequential(
                nn.Conv2d(expanded_ch, c2, 1, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(inplace=True)
            )
        else:
            self.conv = nn.Identity()

    def forward(self, x):
        x = F.pixel_unshuffle(x, self.factor)   # (B, C*F^2, H/F, W/F)
        return self.conv(x)


# ================== 新 WaveletStem ==================
class WaveletStem(nn.Module):
    """
    小波分解 + 阈值去噪 + 逆重建(恢复原尺寸) + SPDConv 下采样压缩。
    输入: (B, in_channels, H, W)
    输出: (B, out_channels, H/2, W/2)
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 64,
                 use_denoise: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_denoise = use_denoise

        # 小波分解滤波器
        self.register_buffer('dec_filters', haar_filters(in_channels))

        # 子带阈值去噪
        if use_denoise:
            self.denoise = SubbandThreshold(in_channels)
        else:
            self.denoise = None

        # SPDConv 下采样：原图 -> 1/2 原图
        self.spd = SPDConv(in_channels, out_channels, factor=2, compress=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape

        # ---------- 1. 小波分解 ----------
        coeffs = wavelet_decompose(x, self.dec_filters)   # (B, 4*C, H/2, W/2)
        B, C4, H2, W2 = coeffs.shape
        C = C4 // 4
        coeffs = coeffs.view(B, C, 4, H2, W2)

        # ---------- 2. 子带阈值去噪 ----------
        if self.denoise is not None:
            coeffs = self.denoise(coeffs)                 # (B, C, 4, H2, W2)

        # ---------- 3. 逆 Haar 重建 -> 恢复原尺寸 ----------
        # 将四个子带合并为 (B, 4*C, H/2, W/2) 后 pixel_shuffle
        combined = coeffs.reshape(B, C * 4, H2, W2)      # (B, 4C, H/2, W/2)
        up = F.pixel_shuffle(combined, 2)                 # (B, C, H, W)

        # ---------- 4. SPD 下采样并压缩通道 ----------
        out = self.spd(up)                                # (B, out_channels, H/2, W/2)
        return out


# ================== 测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 模拟输入 (RGB 640x640)
    x = torch.randn(2, 3, 640, 640).to(device)

    # 创建模型
    model = WaveletStem(in_channels=3, out_channels=128, use_denoise=True).to(device)
    model.eval()

    with torch.no_grad():
        out = model(x)

    print(f"Input shape : {x.shape}")
    print(f"Output shape: {out.shape}")   # 期望 (2, 128, 320, 320)
    expected_shape = (2, 128, 320, 320)
    assert out.shape == expected_shape, f"Shape mismatch! Got {out.shape}"
    print("✅ Output shape verified.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")