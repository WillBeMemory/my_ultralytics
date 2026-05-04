import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


class IWConv(nn.Module):
    """
    重要性加权特征精炼模块（C3k2 风格，串行增强版）
    输入：(B, c1, H, W) → 输出：(B, c2, H, W)

    流程：
        1. 1×1 卷积通道扩增至 2*c_ (c_ = int(c2 * e))
        2. chunk 分两路：一路恒等映射，另一路进行重要性增强
        3. 增强分支：
            - 基础卷积：DWConv (轻量，全图执行)
            - 增强卷积：标准 3×3，全图执行，但输出被重要性权重调制
            - 重要性权重：由输入自身预测 (Conv1x1 + Sigmoid)
        4. 拼接直连与精炼部分后投影输出

    参数:
        c1, c2 : 输入/输出通道数
        n      : 占位参数（兼容框架）
        k      : 卷积核大小，默认 3
        e      : 扩展比，决定隐藏通道 c_ = int(c2 * e)，默认 0.5
    """
    def __init__(self, c1, c2, n=1, k=3, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, 2 * c_, 1, 1)

        # 基础轻量分支 (DWConv + PW)
        self.branch_base = nn.Sequential(
            nn.Conv2d(c_, c_, k, padding=k//2, groups=c_, bias=False),
            nn.Conv2d(c_, c_, 1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True)
        )

        # 增强分支 (标准 3×3)
        self.branch_enhance = nn.Sequential(
            nn.Conv2d(c_, c_, k, padding=k//2, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True)
        )

        # 重要性预测器：预测空间权重图 (B, 1, H, W)
        self.importance = nn.Sequential(
            nn.Conv2d(c_, 1, 1, bias=False),
            nn.Sigmoid()
        )

        # 输出投影
        self.cv2 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        y = self.cv1(x)                        # (B, 2*c_, H, W)
        a, b = y.chunk(2, dim=1)               # 各 (B, c_, H, W)

        base = self.branch_base(b)             # 基础输出

        w = self.importance(b)                 # (B, 1, H, W)，空间重要性

        enhance = self.branch_enhance(b)       # 增强输出
        enhance = enhance * w                  # 重要区域放大，非重要区域抑制

        b_out = base + enhance                 # 串行融合

        out = torch.cat([a, b_out], dim=1)     # 拼接直连与精炼
        return self.cv2(out)


# -------------------- 测试 --------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    c1, c2 = 128, 256
    x = torch.randn(2, c1, 40, 40).to(device)
    model = IWConv(c1, c2, e=0.5).to(device)
    y = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"输入: {x.shape} -> 输出: {y.shape}")
    print(f"参数量: {params:,}")