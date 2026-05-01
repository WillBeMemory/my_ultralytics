import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv


class StarBlock(nn.Module):
    """轻量星操作块，输入输出通道可以不同"""
    def __init__(self, c1, c2, k=3, e=0.5, shortcut=False, act=nn.ReLU6(inplace=True)):
        super().__init__()
        hidden = max(1, int(c1 * e))
        self.hidden = hidden
        self.add = shortcut and c1 == c2

        self.dw1 = nn.Conv2d(c1, c1, k, padding=k // 2, groups=c1, bias=False)
        self.pw1 = nn.Conv2d(c1, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act1 = act

        self.dw2 = nn.Conv2d(c1, c1, k, padding=k // 2, groups=c1, bias=False)
        self.pw2 = nn.Conv2d(c1, hidden, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)

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

        out = x1 * x2                     # 星操作
        out = self.fusion(out)
        out = self.out_act(out)
        if self.add:
            out = out + x
        return out


class StarHighFeature(nn.Module):
    """
    用 StarBlock 对 LH 和 HL 子带进行星操作增强，返回正向特征。
    输入：LH (B, C, H, W), HL (B, C, H, W)
    输出：增强后的 LH, HL，通道数不变。
    """
    def __init__(self, c, reduction=4):
        super().__init__()
        # 输入 2*C，输出 2*C（保持每个子带通道数为 C）
        self.star = StarBlock(c * 2, c * 2, k=3, e=0.5, shortcut=False)

    def forward(self, lh, hl):
        combined = torch.cat([lh, hl], dim=1)
        enhanced = self.star(combined)          # (B, 2*C, H, W)
        lh_out, hl_out = combined.chunk(2, dim=1)  # 各 (B, C, H, W)
        return lh_out, hl_out


class WaveletRefine(nn.Module):
    """
    小波域特征精炼 + 下采样模块。
    LL 子带通过 StarBlock 进行星操作增强。
    LH, HL 子带通过 StarHighFeature 进行星操作增强（正向特征提取）。
    HH 子带不做处理，直接保留。

    可通过 keep_subbands 控制是否输出子带堆叠 (默认 False 为 1×1 融合)。

    输入：(B, c1, H, W)
    输出：(B, c2, H/2, W/2)
    """
    def __init__(self, c1, c2, n=None, shortcut=True, g=1, k=3,
                 star_enhance=True, keep_subbands=False):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.star_enhance = star_enhance
        self.keep_subbands = keep_subbands

        # 固定 Haar 分解滤波器
        w_ll = torch.tensor([[1., 1.], [1., 1.]]) / 2.0
        w_lh = torch.tensor([[1., -1.], [1., -1.]]) / 2.0
        w_hl = torch.tensor([[1., 1.], [-1., -1.]]) / 2.0
        w_hh = torch.tensor([[1., -1.], [-1., 1.]]) / 2.0
        haar_init = torch.stack([w_ll, w_lh, w_hl, w_hh], dim=0).unsqueeze(1)  # (4,1,2,2)
        self.register_buffer('haar_filter', haar_init.repeat(c1, 1, 1, 1))

        # LL 子带星操作增强（输入输出通道均为 c1）
        self.ll_star = StarBlock(c1, c1, k=k, e=0.5, shortcut=True, act=nn.ReLU6(inplace=True))

        # LH, HL 星操作增强模块
        if star_enhance:
            self.high_star = StarHighFeature(c1)

        # 子带独立投影（仅 keep_subbands 模式使用）
        if keep_subbands:
            out_sub_ch = c2 // 4
            self.ll_proj = Conv(c1, out_sub_ch, k=1, act=False)
            self.lh_proj = Conv(c1, out_sub_ch, k=1, act=False)
            self.hl_proj = Conv(c1, out_sub_ch, k=1, act=False)
            self.hh_proj = Conv(c1, out_sub_ch, k=1, act=False)
        else:
            # 标准 1×1 融合
            self.fuse = Conv(c1 * 4, c2, k=1, act=True)

    def forward(self, x):
        B, C, H, W = x.shape
        # 处理奇数尺寸
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # 1. Haar 分解
        coeff = F.conv2d(x, self.haar_filter, stride=2, groups=C)
        coeff = coeff.view(B, C, 4, H // 2, W // 2)

        ll = coeff[:, :, 0]
        lh = coeff[:, :, 1]
        hl = coeff[:, :, 2]
        hh = coeff[:, :, 3]

        # 2. LL 子带星操作增强
        ll_out = self.ll_star(ll)                     # (B, C, H/2, W/2)

        # 3. LH, HL 星操作增强（或原样保留）
        if self.star_enhance:
            lh_out, hl_out = self.high_star(lh, hl)   # 各 (B, C, H/2, W/2)
        else:
            lh_out, hl_out = lh, hl

        hh_out = hh

        # 4. 子带投影或全局融合
        if self.keep_subbands:
            ll_out = self.ll_proj(ll_out)
            lh_out = self.lh_proj(lh_out)
            hl_out = self.hl_proj(hl_out)
            hh_out = self.hh_proj(hh_out)
            # 堆叠并展平为 (B, c2, H/2, W/2)
            out = torch.stack([ll_out, lh_out, hl_out, hh_out], dim=2).reshape(B, self.c2, H // 2, W // 2)
        else:
            # 原有 1×1 融合
            fused = torch.stack([ll_out, lh_out, hl_out, hh_out], dim=2).reshape(B, C * 4, H // 2, W // 2)
            out = self.fuse(fused)

        return out


# 测试代码
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    c1, c2 = 128, 256
    x = torch.randn(2, c1, 40, 40).to(device)

    # 测试标准模式
    model = WaveletRefine(c1, c2, star_enhance=True, keep_subbands=False).to(device)
    out = model(x)
    print(f"Standard mode output shape: {out.shape}")  # [2, 256, 20, 20]

    # 测试 keep_subbands 模式
    model_sub = WaveletRefine(c1, c2, star_enhance=True, keep_subbands=True).to(device)
    out_sub = model_sub(x)
    print(f"Subband mode output shape: {out_sub.shape}")  # [2, 256, 20, 20]

    params = sum(p.numel() for p in model.parameters())
    print(f"Total params (standard): {params:,}")