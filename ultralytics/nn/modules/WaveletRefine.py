import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Bottleneck, Conv


# ---------- 通道注意力 ----------
class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(dim=(2, 3), keepdim=True)          # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)           # (B, 1, C)
        y = self.conv(y)                              # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)         # (B, C, 1, 1)
        return x * self.sigmoid(y)


# ---------- 轻量星操作块 ----------
class StarBlock(nn.Module):
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


# ---------- 纹理抑制模块 ----------
class WaveletTextureSuppress(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.star = StarBlock(c * 2, 1, k=3, e=0.5, shortcut=False)
        self.bias = nn.Parameter(torch.tensor(2.0))     # 初始 ≈ 不抑制
        self.sigmoid = nn.Sigmoid()

    def forward(self, lh, hl):
        combined = torch.cat([lh, hl], dim=1)
        logit = self.star(combined) + self.bias.view(1, -1, 1, 1)
        mask = self.sigmoid(logit)
        return lh * mask, hl * mask


class WaveletRefine(nn.Module):
    """
    保持尺寸不变的 Haar 小波特征精炼模块，替代 C3k2。
    通过 Bottleneck 的扩展比 e 控制参数量。

    参数:
        c1, c2          : 输入/输出通道
        n               : LL Bottleneck 重复次数
        e               : Bottleneck 隐藏通道扩展比（默认 0.25）
        k               : 卷积核大小
        texture_suppress: 是否启用 StarBlock 纹理抑制
        use_eca         : LL 后是否添加 ECA
    """
    def __init__(self, c1, c2, n=1, e=0.5, k=3, texture_suppress=True, use_eca=False):
        super().__init__()
        self.c1, self.c2 = c1, c2
        self.texture_suppress = texture_suppress

        # Haar 滤波器
        w_ll = torch.tensor([[1., 1.], [1., 1.]]) / 2.0
        w_lh = torch.tensor([[1., -1.], [1., -1.]]) / 2.0
        w_hl = torch.tensor([[1., 1.], [-1., -1.]]) / 2.0
        w_hh = torch.tensor([[1., -1.], [-1., 1.]]) / 2.0
        haar_init = torch.stack([w_ll, w_lh, w_hl, w_hh], dim=0).unsqueeze(1)  # (4,1,2,2)
        self.register_buffer('haar_filter', haar_init.repeat(c1, 1, 1, 1))

        # LL 精炼：n 个 Bottleneck，隐藏通道 = int(c1 * e)
        self.ll_refine = nn.Sequential(*[
            Bottleneck(c1, c1, shortcut=False, g=1, k=(k, k), e=e) for _ in range(n)
        ])
        self.ll_eca = ECA(c1) if use_eca else nn.Identity()

        # 纹理抑制
        if texture_suppress:
            self.texture_module = WaveletTextureSuppress(c1)

        # 输出投影
        self.out_conv = Conv(c1, c2, k=1, act=True)

    def forward(self, x):
        B, C, H, W = x.shape
        # 填充到偶数
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect') if pad_h or pad_w else x

        # 1. Haar 分解
        coeff = F.conv2d(x_pad, self.haar_filter, stride=2, groups=C)  # (B,4C,H/2,W/2)
        coeff = coeff.view(B, C, 4, *coeff.shape[-2:])
        ll, lh, hl, hh = coeff[:,:,0], coeff[:,:,1], coeff[:,:,2], coeff[:,:,3]

        # 2. 子带处理
        ll = self.ll_refine(ll)
        ll = self.ll_eca(ll)
        if self.texture_suppress:
            lh, hl = self.texture_module(lh, hl)
        # HH 直通

        # 3. 逆 Haar 重建
        stacked = torch.stack([ll, lh, hl, hh], dim=2)          # (B,C,4,h,w)
        combined = stacked.reshape(B, C*4, *stacked.shape[-2:])
        up = F.pixel_shuffle(combined, 2)                       # (B,C,H_pad,W_pad)

        # 4. 裁边 & 投影
        if pad_h or pad_w: up = up[:,:,:H,:W]
        return self.out_conv(up)


# ---------- 测试 ----------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 示例：c1=128, c2=256, e=0.25 (超轻量)
    model = WaveletRefine(128, 256, n=1, e=0.25, texture_suppress=True, use_eca=False).to(device)
    x = torch.randn(2, 128, 40, 40).to(device)
    y = model(x)
    print(f"输入: {x.shape} -> 输出: {y.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 对比 C3k2
    from ultralytics.nn.modules.block import C3k2
    c3k2 = C3k2(128, 256, n=1, e=0.5, shortcut=False).to(device)
    print(f"C3k2 参数量: {sum(p.numel() for p in c3k2.parameters()):,}")