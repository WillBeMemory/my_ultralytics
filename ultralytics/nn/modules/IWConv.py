import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


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
        out = x1 * x2                     # 星操作
        out = self.fusion(out)
        out = self.out_act(out)
        return out + x if self.add else out


# ---------- 基于重要性加权的卷积模块 ----------
class IWConv(nn.Module):
    """
    基于 L2 范数的重要性加权卷积 (Importance Weighted Convolution)
    输入: (B, c1, H, W) → 输出: (B, c2, H, W)

    参数:
        c1, c2     : 输入/输出通道数
        n          : 占位参数（兼容框架）
        k          : 卷积核大小，默认 3
        e          : 隐藏通道扩展比，默认 0.5
        fusion     : 融合方式，'add' 或 'concat'
        use_star_b : 重要分支是否使用 StarBlock（星操作），默认 False
    """
    def __init__(self, c1, c2, n=1, k=3, e=0.5, fusion='add', use_star_b=False):
        super().__init__()
        self.fusion = fusion
        hidden = max(1, int(c1 * e))

        # ---------- 分支 A：普通精炼（DWConv，轻量） ----------
        self.branch_a = nn.Sequential(
            nn.Conv2d(c1, c1, k, padding=k//2, groups=c1, bias=False),  # DWConv
            nn.Conv2d(c1, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True)
        )

        # ---------- 分支 B：加权精炼（可选用 StarBlock 或普通卷积） ----------
        if use_star_b:
            # 星操作增强重要区域
            self.branch_b = StarBlock(c1, hidden, k=k, e=0.5, shortcut=False)
        else:
            # 标准卷积增强
            self.branch_b = nn.Sequential(
                nn.Conv2d(c1, hidden, k, padding=k//2, bias=False),
                nn.BatchNorm2d(hidden),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.SiLU(inplace=True)
            )

        # ---------- 融合输出 ----------
        if fusion == 'concat':
            self.out_conv = Conv(hidden * 2, c2, k=1, act=True)
        else:
            self.out_conv = Conv(hidden, c2, k=1, act=True)

    def _compute_importance(self, x):
        """计算重要性权重图 (B, 1, H, W)"""
        importance = torch.norm(x, p=2, dim=1, keepdim=True)
        mean = F.avg_pool2d(importance, kernel_size=5, stride=1, padding=2)
        importance = importance / (mean + 1e-6)
        return importance.sigmoid()

    def forward(self, x):
        weight = self._compute_importance(x)   # (B, 1, H, W)

        out_a = self.branch_a(x)               # (B, hidden, H, W)

        # 对重要区域先加权调制，再进行特征增强
        x_weighted = x * weight
        out_b = self.branch_b(x_weighted)      # (B, hidden, H, W)

        if self.fusion == 'concat':
            out = torch.cat([out_a, out_b], dim=1)
        else:
            out = out_a + out_b

        return self.out_conv(out)


# -------------------- 测试 --------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for use_star in [False, True]:
        model = IWConv(128, 256, n=1, k=3, e=0.5, fusion='add', use_star_b=use_star).to(device)
        x = torch.randn(2, 128, 40, 40).to(device)
        y = model(x)
        print(f"use_star_b={use_star}, 输入: {x.shape} -> 输出: {y.shape}, "
              f"参数量: {sum(p.numel() for p in model.parameters()):,}")