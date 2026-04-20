import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import sys
sys.path.append(r'D:\Study\PostGraduate\YOLO_ultralytics\ultralytics')  # 修改为你的实际路径

from .block import Conv


_all_ = ["SPAPF"]
class SPAPF(nn.Module):
    """
    Spatial Pyramid Adaptive Pooling Fusion
    通过可学习权重自适应融合平均池化和最大池化结果
    参考论文：Semantic-guided and multi-dimensional feature perception for target detection in UAV aerial images[citation:2]
    """

    def __init__(self, c1, c2, k=5, reduction=4):
        """
        Args:
            c1: 输入通道数
            c2: 输出通道数
            k: 池化核大小（与SPPF保持一致，默认为5）
            reduction: 中间层通道压缩比例
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels（与原SPPF保持一致）

        # 1×1卷积降维
        self.cv1 = Conv(c1, c_, 1, 1)

        # 池化层（保持与SPPF相同的kernel_size和padding）
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        # 可学习权重生成网络 - 用于自适应融合
        # 输入：拼接后的多尺度特征，输出：每个尺度对应的权重掩码
        self.weight_net = nn.Sequential(
            # 输入通道：c_ * 4（原始 + 三次池化结果）
            nn.Conv2d(c_ * 4, c_ * 4 // reduction, 3, padding=1),
            nn.BatchNorm2d(c_ * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_ * 4 // reduction, 4, 3, padding=1),  # 输出4个权重（对应4个尺度）
            nn.Sigmoid()  # 将权重限制在0~1之间
        )

        # 输出卷积（与原SPPF一致）
        self.cv2 = Conv(c_ * 4, c2, 1, 1)

    def forward(self, x):
        """前向传播"""
        # 1. 初始降维
        x = self.cv1(x)  # (B, c_, H, W)

        # 2. 串联池化（与原SPPF相同，生成多尺度特征）
        y0 = x  # 原始特征
        y1 = self.pool(y0)  # 第一次池化
        y2 = self.pool(y1)  # 第二次池化
        y3 = self.pool(y2)  # 第三次池化

        # 3. 拼接所有尺度特征
        concat_features = torch.cat([y0, y1, y2, y3], dim=1)  # (B, c_*4, H, W)

        # 4. 生成自适应权重掩码
        weights = self.weight_net(concat_features)  # (B, 4, H, W)
        # 将权重拆分到每个尺度
        w0, w1, w2, w3 = weights.chunk(4, dim=1)  # 每个形状 (B, 1, H, W)

        # 5. 加权融合
        # 原始SPPF是直接拼接，这里改为加权融合后再拼接（保持与原SPPF输出通道一致）
        fused_features = torch.cat([
            y0 * w0.expand_as(y0),  # 加权后的原始特征
            y1 * w1.expand_as(y1),  # 加权后的1次池化
            y2 * w2.expand_as(y2),  # 加权后的2次池化
            y3 * w3.expand_as(y3),  # 加权后的3次池化
        ], dim=1)  # (B, c_*4, H, W)

        # 6. 输出卷积
        return self.cv2(fused_features)


if __name__ == "__main__":
    print("=" * 50)
    print("测试 SPAPF 模块")
    print("=" * 50)

    # 测试不同输入尺寸
    test_configs = [
        (1, 512, 32, 32),  # batch=1, channels=512, h=32, w=32
        (2, 256, 64, 64),  # batch=2, channels=256, h=64, w=64
        (4, 1024, 16, 16),  # batch=4, channels=1024, h=16, w=16
    ]

    for i, (b, c, h, w) in enumerate(test_configs):
        print(f"\n[测试{i + 1}] 输入形状: ({b}, {c}, {h}, {w})")
        x = torch.randn(b, c, h, w).requires_grad_(True)

        # 实例化模块（输出通道与输入相同）
        model = SPAPF(c1=c, c2=c, k=5, reduction=4)
        out = model(x)

        print(f"输出形状: {out.shape}")
        assert out.shape == (b, c, h, w), f"输出形状错误: {out.shape}"

        # 测试梯度
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "输入梯度不存在"
        print("✓ 梯度反向传播正常")

    print("\n" + "=" * 50)
    print("所有测试通过！")
