import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================== ECA 模块 ==================
class ECA(nn.Module):
    """Efficient Channel Attention (ECA) 模块。"""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # 自适应计算一维卷积核大小
        t = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = t if t % 2 else t + 1  # 保证奇数
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        y = x.mean(dim=(2, 3), keepdim=True)   # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)    # (B, 1, C)
        y = self.conv(y)                       # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        y = self.sigmoid(y)
        return x * y


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
    def __init__(self, channels: int, init_ll_alpha: float = 0.8):
        super().__init__()
        self.channels = channels
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


# ================== WaveletStem + ECA ==================
class WaveletStem(nn.Module):
    """
    带 ECA 增强的 WaveletStem。
    流程：小波分解 → 子带去噪 → 中间卷积 → ECA → 小波分解 → 输出
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 128, use_denoise: bool = True):
        super().__init__()
        assert out_channels % 4 == 0, "out_channels 必须能被 4 整除"
        mid_channels = out_channels // 4   # 中间通道数，例如 32

        self.in_channels = in_channels
        self.use_denoise = use_denoise

        # 小波滤波器
        self.register_buffer('dec_filters_1', haar_filters(in_channels))
        self.register_buffer('dec_filters_2', haar_filters(mid_channels))

        # 第一次去噪
        self.denoise1 = SubbandDenoise(channels=in_channels) if use_denoise else None

        # 原中间卷积块
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels * 4, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        )

        # 新增：ECA 通道注意力
        self.eca = ECA(mid_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- 第一次分解 + 去噪 ----
        _, _, H, W = x.shape
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        x = wavelet_decompose(x, self.dec_filters_1)   # (B, 12, H/2, W/2)
        B, C4, H2, W2 = x.shape
        C1 = C4 // 4
        x = x.view(B, C1, 4, H2, W2)

        if self.denoise1 is not None:
            x = self.denoise1(x)
        x = x.view(B, C1 * 4, H2, W2)

        # ---- 中间卷积 ----
        x = self.conv_block(x)

        # ---- ECA 通道增强 ----
        x = self.eca(x)

        # ---- 第二次小波分解 ----
        _, _, H2_new, W2_new = x.shape
        pad_h2 = (2 - H2_new % 2) % 2
        pad_w2 = (2 - W2_new % 2) % 2
        if pad_h2 > 0 or pad_w2 > 0:
            x = F.pad(x, (0, pad_w2, 0, pad_h2), mode='reflect')

        x = wavelet_decompose(x, self.dec_filters_2)   # (B, out_channels, H/4, W/4)
        return x


# ================== 测试 ==================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WaveletStem(in_channels=3, out_channels=128, use_denoise=True).to(device)
    x = torch.randn(2, 3, 640, 640).to(device)
    out = model(x)
    print(f"Input  shape: {x.shape}")
    print(f"Output shape: {out.shape}")   # 期望 (2, 128, 160, 160)
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    # 单独测试 ECA 模块
    eca_test = ECA(64).to(device)
    test_tensor = torch.randn(2, 64, 32, 32).to(device)
    out_eca = eca_test(test_tensor)
    print(f"ECA test: input {test_tensor.shape} -> output {out_eca.shape} (should be same)")