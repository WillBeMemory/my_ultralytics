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


# ================== 子带去噪 ==================
class SubbandDenoise(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.ll_alpha = nn.Parameter(torch.tensor(2.0))
        self.lh_hl_thr = nn.Parameter(torch.tensor(-4.0))
        self.hh_thr = nn.Parameter(torch.full((channels,), -4.0))

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        ll = coeffs[:, :, 0, :, :]
        lh = coeffs[:, :, 1, :, :]
        hl = coeffs[:, :, 2, :, :]
        hh = coeffs[:, :, 3, :, :]

        alpha = torch.sigmoid(self.ll_alpha)
        ll = ll * alpha

        thr_lh_hl = F.softplus(self.lh_hl_thr)
        lh = torch.sign(lh) * F.relu(torch.abs(lh) - thr_lh_hl)
        hl = torch.sign(hl) * F.relu(torch.abs(hl) - thr_lh_hl)

        thr_hh = F.softplus(self.hh_thr).view(1, -1, 1, 1)
        hh = torch.sign(hh) * F.relu(torch.abs(hh) - thr_hh)

        return torch.stack([ll, lh, hl, hh], dim=2)


# ================== 修改后的 WaveletStem (一次分解+去噪) ==================
class WaveletStem(nn.Module):
    """
    一次 Haar 小波分解 + 去噪的 Stem 模块。
    输入: (B, in_channels, H, W)
    输出: (B, out_channels, H/2, W/2)
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 128, use_denoise: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_denoise = use_denoise

        # 第一次小波分解滤波器
        self.register_buffer('dec_filters', haar_filters(in_channels))

        # 子带去噪
        if use_denoise:
            self.denoise = SubbandDenoise(channels=in_channels)
        else:
            self.denoise = None

        # 中间卷积块：直接输出目标通道数，不再进行第二次分解
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 处理奇数尺寸
        _, _, H, W = x.shape
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # 第一次小波分解
        x = wavelet_decompose(x, self.dec_filters)       # (B, 4*C_in, H/2, W/2)
        B, C4, H2, W2 = x.shape
        C1 = C4 // 4
        x = x.view(B, C1, 4, H2, W2)

        # 去噪
        if self.denoise is not None:
            x = self.denoise(x)
        x = x.view(B, C1 * 4, H2, W2)                    # 展平回 (B, C1*4, H2, W2)

        # 中间卷积，直接输出目标通道
        x = self.conv_block(x)                            # (B, out_channels, H/2, W/2)
        return x


# ================== 测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = WaveletStem(in_channels=3, out_channels=256, use_denoise=True).to(device)
    x = torch.randn(2, 3, 640, 640).to(device)
    out = model(x)
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {out.shape}")   # 期望 (2, 256, 320, 320)
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")