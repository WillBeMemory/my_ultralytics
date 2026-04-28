import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv, Bottleneck   # 直接使用标准瓶颈


# ==================== StarBlock (轻量星操作块) ====================
class StarBlock(nn.Module):
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


# ==================== 小波纹理抑制模块 ====================
class WaveletTextureSuppress(nn.Module):
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


# ==================== WaveletRefine 模块 ====================
class WaveletRefine(nn.Module):
    """
    小波域特征精炼 + 下采样模块，替代 C3k2 + HWD。
    输入：(B, C, H, W)
    输出：(B, C2, H/2, W/2)

    参数:
        c1               : 输入通道数
        c2               : 输出通道数
        n                : LL 子带精炼的瓶颈数量（标准 Bottleneck）
        shortcut         : 瓶颈内部是否使用残差连接
        g                : 分组卷积组数
        k                : 卷积核大小
        texture_suppress : 是否启用 LH/HL 纹理抑制
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, k=3,
                 texture_suppress=True):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.texture_suppress = texture_suppress

        # 固定 Haar 分解滤波器
        w_ll = torch.tensor([[1., 1.], [1., 1.]]) / 2.0
        w_lh = torch.tensor([[1., -1.], [1., -1.]]) / 2.0
        w_hl = torch.tensor([[1., 1.], [-1., -1.]]) / 2.0
        w_hh = torch.tensor([[1., -1.], [-1., 1.]]) / 2.0
        haar_init = torch.stack([w_ll, w_lh, w_hl, w_hh], dim=0).unsqueeze(1)  # (4,1,2,2)
        self.register_buffer('haar_filter', haar_init.repeat(c1, 1, 1, 1))

        # LL 子带精炼：多个标准瓶颈串联（无内部分支）
        self.ll_refine = nn.Sequential(*[
            Bottleneck(c1, c1, shortcut, g, k=(k, k), e=1.0) for _ in range(n)
        ])

        # 纹理抑制分支
        if texture_suppress:
            self.texture_module = WaveletTextureSuppress(c1)

        # 最终 1x1 融合
        self.fuse = Conv(c1 * 4, c2, k=1, act=True)

    def forward(self, x):
        B, C, H, W = x.shape
        # 处理奇数尺寸
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # 1. Haar 分解 -> (B, 4*C, H/2, W/2)
        coeff = F.conv2d(x, self.haar_filter, stride=2, groups=C)
        coeff = coeff.view(B, C, 4, H // 2, W // 2)

        ll = coeff[:, :, 0]
        lh = coeff[:, :, 1]
        hl = coeff[:, :, 2]
        hh = coeff[:, :, 3]

        # 2. 子带精炼与抑制
        ll_out = self.ll_refine(ll)          # LL 整体经过瓶颈，无内部直连分支
        if self.texture_suppress:
            lh_out, hl_out = self.texture_module(lh, hl)
        else:
            lh_out, hl_out = lh, hl
        hh_out = hh

        # 3. 堆叠并压缩
        fused = torch.stack([ll_out, lh_out, hl_out, hh_out], dim=2)
        fused = fused.reshape(B, C * 4, H // 2, W // 2)
        out = self.fuse(fused)               # (B, C2, H/2, W/2)
        return out


# ==================== 测试 ====================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    c1, c2 = 128, 256
    x = torch.randn(2, c1, 40, 40).to(device)

    configs = [
        ("no suppress, 1 bottleneck", dict(n=1, texture_suppress=False)),
        ("suppress, 1 bottleneck",    dict(n=1, texture_suppress=True)),
        ("no suppress, 2 bottlenecks", dict(n=2, texture_suppress=False)),
        ("suppress, 2 bottlenecks",  dict(n=2, texture_suppress=True)),
    ]

    for desc, kwargs in configs:
        model = WaveletRefine(c1, c2, **kwargs).to(device)
        out = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"--- {desc} ---")
        print(f"  Input : {x.shape}")
        print(f"  Output: {out.shape}  (expected [2, {c2}, 20, 20])")
        print(f"  Params: {params:,}\n")
        assert out.shape == (2, c2, 20, 20)