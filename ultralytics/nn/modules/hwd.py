import torch
import torch.nn as nn
import torch.nn.functional as F


def haar_filters_groups(in_channels: int):
    """生成形状 (in_channels*4, 1, 2, 2)，保证每组内顺序为 [LL, LH, HL, HH]"""
    dec_lo = torch.tensor([1 / 2, 1 / 2])
    dec_hi = torch.tensor([1 / 2, -1 / 2])
    base_LL = (dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    base_LH = (dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    base_HL = (dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    base_HH = (dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    per_channel = torch.cat([base_LL, base_LH, base_HL, base_HH], dim=1)  # (1,4,2,2)
    per_channel = per_channel.repeat(in_channels, 1, 1, 1)  # (C,4,2,2)
    return per_channel.reshape(in_channels * 4, 1, 2, 2)  # (4C,1,2,2)


def wt_decompose_level(x: torch.Tensor):
    """输入 (B, C, H, W)，返回四个子带 (B, C, H/2, W/2)"""
    B, C, H, W = x.shape
    filters = haar_filters_groups(C).to(device=x.device, dtype=x.dtype)
    coeffs = F.conv2d(x, filters, stride=2, groups=C, padding=0)  # (B, 4C, H/2, W/2)
    coeffs = coeffs.view(B, C, 4, H // 2, W // 2)
    LL = coeffs[:, :, 0, :, :]
    LH = coeffs[:, :, 1, :, :]
    HL = coeffs[:, :, 2, :, :]
    HH = coeffs[:, :, 3, :, :]
    return LL, LH, HL, HH


class HWD(nn.Module):
    def __init__(self, c1: int, c2: int, enhance_ll: bool = True):
        super().__init__()
        assert c2 % 4 == 0, f"c2 must be divisible by 4, got {c2}"
        self.c1 = c1
        self.c2 = c2
        self.enhance_ll = enhance_ll
        ch_per_band = c2 // 4

        self.ll_proj = nn.Conv2d(c1, ch_per_band, 1, bias=False)
        self.lh_proj = nn.Conv2d(c1, ch_per_band, 1, bias=False)
        self.hl_proj = nn.Conv2d(c1, ch_per_band, 1, bias=False)
        self.hh_proj = nn.Conv2d(c1, ch_per_band, 1, bias=False)

        if self.enhance_ll:
            self.ll_enhance = nn.Sequential(
                nn.Conv2d(ch_per_band, ch_per_band, 3, padding=1, bias=False),  # 标准卷积
                nn.BatchNorm2d(ch_per_band),
                nn.SiLU(inplace=True)
            )
        else:
            self.ll_enhance = nn.Identity()

        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        LL, LH, HL, HH = wt_decompose_level(x)

        ll = self.ll_proj(LL)
        lh = self.lh_proj(LH)
        hl = self.hl_proj(HL)
        hh = self.hh_proj(HH)

        if self.enhance_ll:
            ll = ll + self.ll_enhance(ll)  # 残差连接

        out = torch.cat([ll, lh, hl, hh], dim=1)
        return self.act(self.bn(out))

# ================== 简单 main 测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 测试增强版 HWD
    hwd = HWD(c1=3, c2=64, enhance_ll=True).to(device)
    x = torch.randn(2, 3, 256, 256).to(device)
    out = hwd(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")  # (2, 64, 128, 128)
    expected = (2, 64, 128, 128)
    assert out.shape == expected, f"Shape mismatch: {out.shape}"
    print("✅ Output shape verified.")

    # 参数量
    total_params = sum(p.numel() for p in hwd.parameters())
    print(f"Total parameters (LL enhanced): {total_params:,}")

    # 测试不增强版本
    hwd_light = HWD(3, 64, enhance_ll=False).to(device)
    out_l = hwd_light(x)
    print(f"Light version output shape: {out_l.shape}")
    print("✅ All tests passed.")