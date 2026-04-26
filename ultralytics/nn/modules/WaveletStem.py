import torch
import torch.nn as nn
import torch.nn.functional as F


# ================== StarBlock ==================
class StarBlock(nn.Module):
    """极简星操作块，输入输出通道可以不同"""
    def __init__(self, c1, c2, k=3, e=0.5, act=nn.ReLU6(inplace=True)):
        super().__init__()
        hidden = max(1, int(c1 * e))
        self.hidden = hidden
        # 分支1
        self.dw1 = nn.Conv2d(c1, c1, k, padding=k//2, groups=c1, bias=False)
        self.pw1 = nn.Conv2d(c1, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act1 = act
        # 分支2
        self.dw2 = nn.Conv2d(c1, c1, k, padding=k//2, groups=c1, bias=False)
        self.pw2 = nn.Conv2d(c1, hidden, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        # 融合并映射到目标通道
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden, c2, 1, bias=False),
            nn.BatchNorm2d(c2)
        )
        self.out_act = act

    def forward(self, x):
        x1 = self.dw1(x)
        x1 = self.pw1(x1)
        x1 = self.bn1(x1)
        x1 = self.act1(x1)

        x2 = self.dw2(x)
        x2 = self.pw2(x2)
        x2 = self.bn2(x2)

        out = x1 * x2                     # ★ 星操作
        out = self.fusion(out)
        out = self.out_act(out)
        return out


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


# ================== WaveletStem（StarBlock 增强版） ==================
class WaveletStem(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 128, use_denoise: bool = True):
        super().__init__()
        assert out_channels % 4 == 0, "out_channels 必须能被 4 整除"
        mid_channels = out_channels // 4

        self.in_channels = in_channels
        self.use_denoise = use_denoise

        # 小波滤波器
        self.register_buffer('dec_filters_1', haar_filters(in_channels))
        self.register_buffer('dec_filters_2', haar_filters(mid_channels))

        # 第一次去噪
        self.denoise1 = SubbandDenoise(channels=in_channels) if use_denoise else None

        # ★ 用 StarBlock 替代原中间卷积
        self.star_block = StarBlock(in_channels * 4, mid_channels, k=3, e=0.5)

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

        # ---- StarBlock 增强（替代中间卷积） ----
        x = self.star_block(x)                         # (B, mid_channels, H/2, W/2)

        # ---- 第二次分解 ----
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
    print(f"Output shape: {out.shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")