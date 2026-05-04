import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv


# ---------- 辅助模块 ----------
class StarBlock(nn.Module):
    """轻量星操作块"""
    def __init__(self, c1, c2, k=3, e=0.5, shortcut=False, act=nn.ReLU6(inplace=True)):
        super().__init__()
        hidden = max(1, int(c1 * e))
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
        x1 = self.dw1(x); x1 = self.pw1(x1); x1 = self.bn1(x1); x1 = self.act1(x1)
        x2 = self.dw2(x); x2 = self.pw2(x2); x2 = self.bn2(x2)
        out = x1 * x2
        out = self.fusion(out)
        out = self.out_act(out)
        return out + x if self.add else out


class WaveletTextureSuppress(nn.Module):
    """小波纹理抑制"""
    def __init__(self, c):
        super().__init__()
        self.star = StarBlock(c * 2, 1, k=3, e=0.5, shortcut=False)
        self.bias = nn.Parameter(torch.tensor(2.0))
        self.sigmoid = nn.Sigmoid()

    def forward(self, lh, hl):
        combined = torch.cat([lh, hl], dim=1)
        logit = self.star(combined) + self.bias.view(1, -1, 1, 1)
        mask = self.sigmoid(logit)
        return lh * mask, hl * mask


class ECA(nn.Module):
    """高效通道注意力"""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        import math
        t = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(dim=(2, 3), keepdim=True)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


# ====================== WaveletRefine ======================
class WaveletRefine(nn.Module):
    """
    小波域特征精炼 + 下采样模块。
    输入: (B, c1, H, W) → 输出: (B, c2, H/2, W/2)

    子带处理:
        - LL: 双重标准卷积
        - HH: 深度可分离卷积 (DWConv + 1x1)
        - LH, HL: 纹理抑制
        - 四个子带拼接 -> ECA -> 1x1 融合卷积
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, k=3,
                 texture_suppress=True):
        super().__init__()
        self.c1, self.c2 = c1, c2
        self.texture_suppress = texture_suppress

        # ---- 固定 Haar 分解滤波器 ----
        w_ll = torch.tensor([[1., 1.], [1., 1.]]) / 2.0
        w_lh = torch.tensor([[1., -1.], [1., -1.]]) / 2.0
        w_hl = torch.tensor([[1., 1.], [-1., -1.]]) / 2.0
        w_hh = torch.tensor([[1., -1.], [-1., 1.]]) / 2.0
        haar_init = torch.stack([w_ll, w_lh, w_hl, w_hh], dim=0).unsqueeze(1)  # (4,1,2,2)
        self.register_buffer('haar_filter', haar_init.repeat(c1, 1, 1, 1))

        # ---- LL 子带：双重标准卷积 ----
        self.ll_refine = nn.Sequential(
            nn.Conv2d(c1, c1, k, padding=k//2, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            # nn.Conv2d(c1, c1, k, padding=k//2, bias=False),
            # nn.BatchNorm2d(c1),
            # nn.SiLU(inplace=True)
        )

        # ---- HH 子带：深度可分离卷积 ----
        self.hh_conv = nn.Sequential(
            nn.Conv2d(c1, c1, k, padding=k//2, groups=c1, bias=False),  # DWConv
            nn.Conv2d(c1, c1, 1, bias=False),                          # 1×1
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True)
        )

        # ---- LH, HL 纹理抑制 ----
        if texture_suppress:
            self.texture_module = WaveletTextureSuppress(c1)

        # ---- 融合（ECA + 1x1 卷积） ----
        self.eca = ECA(c1 * 4)
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
        ll, lh, hl, hh = coeff[:, :, 0], coeff[:, :, 1], coeff[:, :, 2], coeff[:, :, 3]

        # 2. 子带处理
        ll_out = self.ll_refine(ll)
        hh_out = self.hh_conv(hh)
        if self.texture_suppress:
            lh_out, hl_out = self.texture_module(lh, hl)
        else:
            lh_out, hl_out = lh, hl

        # 3. 直接拼接四个子带 (B, 4*c1, H/2, W/2)
        out = torch.cat([ll_out, lh_out, hl_out, hh_out], dim=1)

        # 4. ECA 通道增强
        out = self.eca(out)

        # 5. 1x1 融合卷积
        out = self.fuse(out)
        return out


# ====================== 测试 ======================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    c1, c2 = 128, 256
    x = torch.randn(2, c1, 40, 40).to(device)
    model = WaveletRefine(c1, c2, texture_suppress=True).to(device)
    y = model(x)
    print(f"输入: {x.shape} -> 输出: {y.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")