import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv


# ---------- 轻量星操作块 ----------
class StarBlock(nn.Module):
    """
    轻量星操作块，输入输出通道可以不同。
    内部使用深度可分离卷积（DWConv）降低参数量。
    """
    def __init__(self, c1, c2, k=3, e=0.5, shortcut=False, act=nn.ReLU6(inplace=True)):
        super().__init__()
        hidden = max(1, int(c1 * e))
        self.add = shortcut and c1 == c2

        # 分支1：DWConv -> 1x1 -> BN -> ReLU6
        self.dw1 = nn.Conv2d(c1, c1, k, padding=k//2, groups=c1, bias=False)
        self.pw1 = nn.Conv2d(c1, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act1 = act

        # 分支2：DWConv -> 1x1 -> BN (无激活)
        self.dw2 = nn.Conv2d(c1, c1, k, padding=k//2, groups=c1, bias=False)
        self.pw2 = nn.Conv2d(c1, hidden, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)

        # 融合层：将星操作结果映射到输出通道
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
        if self.add:
            out = out + x
        return out


# ====================== WaveletRefine（替代下采样，星操作增强） ======================
class WaveletRefine(nn.Module):
    """
    使用 Haar 小波分解进行 2 倍下采样，并对四个子带分别用 StarBlock 进行特征增强，
    最后堆叠并 1×1 融合到目标输出通道。

    输入 (B, c1, H, W) → 输出 (B, c2, H/2, W/2)

    参数:
        c1, c2     : 输入/输出通道数
        k          : StarBlock 中的卷积核大小
        e          : StarBlock 隐藏通道扩展比 (默认 0.25，极轻量)
        act        : 激活函数
    """
    def __init__(self, c1, c2, k=3, e=0.25, act=nn.ReLU6(inplace=True)):
        super().__init__()
        self.c1, self.c2 = c1, c2

        # ---------- 固定 Haar 分解滤波器 ----------
        w_ll = torch.tensor([[1., 1.], [1., 1.]]) / 2.0
        w_lh = torch.tensor([[1., -1.], [1., -1.]]) / 2.0
        w_hl = torch.tensor([[1., 1.], [-1., -1.]]) / 2.0
        w_hh = torch.tensor([[1., -1.], [-1., 1.]]) / 2.0
        haar_init = torch.stack([w_ll, w_lh, w_hl, w_hh], dim=0).unsqueeze(1)  # (4, 1, 2, 2)
        self.register_buffer('haar_filter', haar_init.repeat(c1, 1, 1, 1))

        # ---------- 四个子带的星操作增强（各自独立） ----------
        self.ll_star = StarBlock(c1, c1, k=k, e=e, act=act)
        self.lh_star = StarBlock(c1, c1, k=k, e=e, act=act)
        self.hl_star = StarBlock(c1, c1, k=k, e=e, act=act)
        self.hh_star = StarBlock(c1, c1, k=k, e=e, act=act)

        # ---------- 最终融合（4×c1 → c2） ----------
        self.fuse = Conv(c1 * 4, c2, k=1, act=True)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. Haar 分解
        coeff = F.conv2d(x, self.haar_filter, stride=2, groups=C)  # (B, 4C, H/2, W/2)
        coeff = coeff.view(B, C, 4, H//2, W//2)
        ll, lh, hl, hh = coeff[:, :, 0], coeff[:, :, 1], coeff[:, :, 2], coeff[:, :, 3]

        # 2. 各子带星操作增强
        ll = self.ll_star(ll)
        lh = self.lh_star(lh)
        hl = self.hl_star(hl)
        hh = self.hh_star(hh)

        # 3. 堆叠 → 融合
        fused = torch.stack([ll, lh, hl, hh], dim=2).reshape(B, C * 4, H//2, W//2)
        out = self.fuse(fused)
        return out


# ====================== 测试 ======================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WaveletRefine(128, 256, k=3, e=0.25).to(device)
    x = torch.randn(2, 128, 40, 40).to(device)
    y = model(x)
    print(f"输入: {x.shape} -> 输出: {y.shape}")  # [2,256,20,20]
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")