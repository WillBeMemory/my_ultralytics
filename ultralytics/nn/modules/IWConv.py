import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


class IWConv(nn.Module):
    """
    基于 L2 范数的重要性加权卷积 (Importance Weighted Convolution)
    输入: (B, c1, H, W) → 输出: (B, c2, H, W)

    参数:
        c1, c2 : 输入/输出通道数
        n      : 占位参数（兼容 C3k2，不实际使用）
        k      : 深度可分离卷积核大小，默认 3
        e      : 隐藏通道扩展比，默认 0.5
        fusion : 融合方式，'add' 或 'concat'
    """
    def __init__(self, c1, c2, n=1, k=3, e=0.5, fusion='add'):
        super().__init__()
        self.fusion = fusion
        hidden = max(1, int(c1 * e))

        # ---------- 分支 A：普通精炼 ----------
        self.branch_a = nn.Sequential(
            nn.Conv2d(c1, c1, k, padding=k//2, groups=c1, bias=False),  # DWConv
            nn.Conv2d(c1, hidden, 1, bias=False),                       # 1x1 压缩
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True)
        )

        # ---------- 分支 B：加权精炼 ----------
        self.branch_b = nn.Sequential(
            nn.Conv2d(c1, c1, k, padding=k//2, groups=c1, bias=False),
            nn.Conv2d(c1, hidden, 1, bias=False),
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
        # L2 范数
        importance = torch.norm(x, p=2, dim=1, keepdim=True)
        # 局部对比度归一化 (5x5 邻域)
        mean = F.avg_pool2d(importance, kernel_size=5, stride=1, padding=2)
        importance = importance / (mean + 1e-6)
        # Sigmoid 映射到 [0,1]
        return importance.sigmoid()

    def forward(self, x):
        weight = self._compute_importance(x)   # (B, 1, H, W)

        # 分支 A
        out_a = self.branch_a(x)               # (B, hidden, H, W)

        # 分支 B：先对输入施加权重，再卷积
        x_weighted = x * weight
        out_b = self.branch_b(x_weighted)

        # 融合
        if self.fusion == 'concat':
            out = torch.cat([out_a, out_b], dim=1)
        else:
            out = out_a + out_b

        return self.out_conv(out)


# -------------------- 测试 --------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = IWConv(128, 256, n=1, k=3, e=0.25, fusion='add').to(device)
    x = torch.randn(2, 128, 40, 40).to(device)
    y = model(x)
    print(f"输入: {x.shape} -> 输出: {y.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")