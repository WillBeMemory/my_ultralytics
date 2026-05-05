import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


class IWConv(nn.Module):
    """
    重要性加权特征精炼模块（C3k2 风格，串行增强，L2 归一化 + 标准差）
    输入：(B, c1, H, W) → 输出：(B, c2, H, W)

    改进点：
        - 重要性评估：通道 L2 归一化 + 局部标准差，对纹理复杂度敏感且不受绝对能量干扰
        - 增强分支：标准 3×3 卷积，高响应区域获得强精炼
        - 可训练偏置，稳定早期训练
    """
    def __init__(self, c1, c2, n=1, k=3, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, 2 * c_, 1, 1)

        # 基础分支：深度可分离卷积
        self.branch_base = nn.Sequential(
            nn.Conv2d(c_, c_, k, padding=k//2, groups=c_, bias=False),
            nn.Conv2d(c_, c_, 1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True)
        )

        # 增强分支：标准 3×3 卷积
        self.branch_enhance = nn.Sequential(
            nn.Conv2d(c_, c_, k, padding=k//2, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True)
        )

        # 重要性预测器
        self.importance = nn.Sequential(
            nn.Conv2d(1, 1, 1, bias=True),    # 将标准差压缩到 [0,1]
            nn.Sigmoid()
        )
        nn.init.constant_(self.importance[0].bias, -0.5)

        self.cv2 = Conv(2 * c_, c2, 1, 1)

    def _compute_importance(self, x):
        """计算通道归一化标准差重要性图 (B, 1, H, W)"""
        x_norm = F.normalize(x, p=2, dim=1)          # 每个位置的通道向量长度 = 1
        mean = F.avg_pool2d(x_norm, kernel_size=5, stride=1, padding=2)
        sq_mean = F.avg_pool2d(x_norm.pow(2), kernel_size=5, stride=1, padding=2)
        std = (sq_mean - mean.pow(2)).clamp(min=0).sqrt()
        imp = std.mean(dim=1, keepdim=True)          # (B, 1, H, W)
        return imp

    def forward(self, x):
        y = self.cv1(x)
        a, b = y.chunk(2, dim=1)

        base = self.branch_base(b)

        imp = self._compute_importance(b)
        w = self.importance(imp)

        enhance = self.branch_enhance(b)
        enhance = enhance * w

        b_out = base + enhance
        out = torch.cat([a, b_out], dim=1)
        return self.cv2(out)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}\n")

    # 输入参数
    c1, c2 = 128, 256
    x = torch.randn(2, c1, 40, 40).to(device)

    # 创建模型
    model = IWConv(c1, c2, n=1, k=3, e=0.5).to(device)

    # 前向传播
    y = model(x)

    # 打印结果
    print(f"输入形状 : {x.shape}")
    print(f"输出形状 : {y.shape}")
    print(f"参数量   : {sum(p.numel() for p in model.parameters()):,}")

    # 验证输出形状
    expected_shape = (2, c2, 40, 40)
    assert y.shape == expected_shape, f"形状不匹配！期望 {expected_shape}，实际 {y.shape}"
    print("\n✅ 测试通过！输出形状与预期一致。")

