import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv, Bottleneck


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


class WaveletTextureSuppress(nn.Module):
    """用 StarBlock 实现小波子带纹理抑制"""
    def __init__(self, c, reduction=4):
        super().__init__()
        self.star = StarBlock(c * 2, 1, k=3, e=0.5, shortcut=False)
        self.bias = nn.Parameter(torch.tensor(2.0))
        self.sigmoid = nn.Sigmoid()

    def forward(self, lh, hl):
        combined = torch.cat([lh, hl], dim=1)
        logit = self.star(combined) + self.bias.view(1, -1, 1, 1)
        mask = self.sigmoid(logit)
        lh = lh * mask
        hl = hl * mask
        return lh, hl


class WaveletRefine(nn.Module):
    """
    小波域特征精炼 + 下采样模块，替代 C3k2 + HWD。
    LL 子带通过 Bottleneck 温和精炼（保护结构信息）。
    LH, HL 子带通过 WaveletTextureSuppress 进行纹路过滤。
    HH 子带不做处理，直接保留。

    支持 keep_subbands 模式，每个子带独立投影后堆叠输出。

    输入：(B, c1, H, W)
    输出：(B, c2, H/2, W/2) 或 (B, c2, H/2, W/2) 子带堆叠
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, k=3,
                 texture_suppress=True, keep_subbands=False):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.texture_suppress = texture_suppress
        self.keep_subbands = keep_subbands

        # ---------- 固定 Haar 分解滤波器 ----------
        w_ll = torch.tensor([[1., 1.], [1., 1.]]) / 2.0
        w_lh = torch.tensor([[1., -1.], [1., -1.]]) / 2.0
        w_hl = torch.tensor([[1., 1.], [-1., -1.]]) / 2.0
        w_hh = torch.tensor([[1., -1.], [-1., 1.]]) / 2.0
        haar_init = torch.stack([w_ll, w_lh, w_hl, w_hh], dim=0).unsqueeze(1)
        self.register_buffer('haar_filter', haar_init.repeat(c1, 1, 1, 1))

        # ---------- LL 子带精炼：使用 Bottleneck (温和、稳定) ----------
        self.ll_refine = nn.Sequential(*[
            Bottleneck(c1, c1, shortcut, g, k=(k, k), e=1.0) for _ in range(n)
        ])

        # ---------- LH, HL 纹理抑制 ----------
        if texture_suppress:
            self.texture_module = WaveletTextureSuppress(c1)

        # ---------- 子带独立投影 (keep_subbands 模式) ----------
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

        # 2. LL 子带 Bottleneck 精炼
        ll_out = self.ll_refine(ll)                     # (B, C, H/2, W/2)

        # 3. LH, HL 纹理过滤
        if self.texture_suppress:
            lh_out, hl_out = self.texture_module(lh, hl)
        else:
            lh_out, hl_out = lh, hl

        hh_out = hh

        # 4. 子带投影或全局融合
        if self.keep_subbands:
            ll_out = self.ll_proj(ll_out)
            lh_out = self.lh_proj(lh_out)
            hl_out = self.hl_proj(hl_out)
            hh_out = self.hh_proj(hh_out)
            out = torch.stack([ll_out, lh_out, hl_out, hh_out], dim=2).reshape(B, self.c2, H // 2, W // 2)
        else:
            fused = torch.stack([ll_out, lh_out, hl_out, hh_out], dim=2).reshape(B, C * 4, H // 2, W // 2)
            out = self.fuse(fused)

        return out


# ==================== 简单测试 ====================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    c1, c2 = 128, 256
    x = torch.randn(2, c1, 40, 40).to(device)

    # 标准模式
    model = WaveletRefine(c1, c2, n=2, texture_suppress=True).to(device)
    out = model(x)
    print(f"Standard mode: {out.shape}")  # [2, 256, 20, 20]

    # keep_subbands 模式
    model_sub = WaveletRefine(c1, c2, n=2, texture_suppress=True, keep_subbands=True).to(device)
    out_sub = model_sub(x)
    print(f"Subband mode: {out_sub.shape}")  # [2, 256, 20, 20]

    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")