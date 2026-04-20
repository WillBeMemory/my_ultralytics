import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义模块（与项目中保持一致）
class WeightedP2Fusion(nn.Module):
    """
    先压缩 P2 通道，再下采样并与目标特征加权融合。
    Args:
        target_channels: 目标特征（如 P3）的通道数
        p2_channels: 原始 P2 的通道数
        compress_channels: 压缩后的中间通道数（可选，默认为 target_channels//2 或可设定）
    """
    def __init__(self, target_channels, p2_channels, compress_channels=None):
        super().__init__()
        if compress_channels is None:
            compress_channels = max(target_channels // 4, 16)  # 默认压缩到目标通道的 1/4
        # 压缩层
        self.compress = nn.Sequential(
            nn.Conv2d(p2_channels, compress_channels, 1, bias=False),
            nn.BatchNorm2d(compress_channels),
            nn.SiLU(inplace=True)
        )
        # 降维到目标通道（用于后续加权）
        self.reduce = nn.Conv2d(compress_channels, target_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(target_channels)
        self.act = nn.ReLU(inplace=True)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, target, p2):
        # 1. 压缩通道
        x = self.compress(p2)
        # 2. 下采样到目标尺寸
        x_down = F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)
        # 3. 降维到目标通道
        x_down = self.reduce(x_down)
        x_down = self.bn(x_down)
        x_down = self.act(x_down)
        # 4. 加权融合
        weight = torch.sigmoid(self.alpha)
        return (1 - weight) * target + weight * x_down

def test_weighted_p2_fusion():
    print("=" * 50)
    print("测试 WeightedP2Fusion 模块")
    print("=" * 50)

    # 设置随机种子以便复现
    torch.manual_seed(42)

    # 测试用例 1: 基本形状匹配
    print("\n[测试1] 基本形状匹配 (batch=2, target通道=256, P2通道=128, 目标尺寸8x8, P2尺寸16x16)")
    batch = 2
    target_c = 256
    p2_c = 128
    target_h, target_w = 8, 8
    p2_h, p2_w = 16, 16

    # 创建随机输入
    target_feat = torch.randn(batch, target_c, target_h, target_w)
    p2_feat = torch.randn(batch, p2_c, p2_h, p2_w)

    # 实例化模块
    model = WeightedP2Fusion(target_channels = target_c,p2_channels=p2_c)
    print("模块初始化成功。")
    print(f"目标特征形状: {target_feat.shape}")
    print(f"P2特征形状: {p2_feat.shape}")

    # 前向传播
    output = model(target_feat, p2_feat)
    print(f"输出特征形状: {output.shape}")

    # 断言输出形状与目标相同
    assert output.shape == target_feat.shape, f"输出形状 {output.shape} 与目标形状 {target_feat.shape} 不一致！"
    print("✓ 输出形状正确。")

    # 测试可学习权重
    print(f"\n初始 alpha 值: {model.alpha.item():.4f} (sigmoid后权重: {torch.sigmoid(model.alpha).item():.4f})")
    # 检查 alpha 是否 requires_grad
    assert model.alpha.requires_grad, "alpha 参数未设置 requires_grad！"
    print("✓ alpha 是可学习参数。")

    # 测试梯度流
    loss = output.sum()
    loss.backward()
    assert model.reduce.weight.grad is not None, "reduce 卷积的梯度为 None！"
    assert model.bn.weight.grad is not None, "BN 的梯度为 None！"
    assert model.alpha.grad is not None, "alpha 的梯度为 None！"
    print("✓ 梯度反向传播正常。")

    # 测试用例 2: 不同通道组合
    print("\n[测试2] 不同通道组合 (target_c=512, p2_c=128, 目标尺寸16x16, P2尺寸32x32)")
    target_c2 = 512
    p2_c2 = 128
    target_h2, target_w2 = 16, 16
    p2_h2, p2_w2 = 32, 32

    target_feat2 = torch.randn(batch, target_c2, target_h2, target_w2)
    p2_feat2 = torch.randn(batch, p2_c2, p2_h2, p2_w2)

    model2 = WeightedP2Fusion(target_channels=target_c2, p2_channels=p2_c2)
    output2 = model2(target_feat2, p2_feat2)
    print(f"目标特征形状: {target_feat2.shape}")
    print(f"P2特征形状: {p2_feat2.shape}")
    print(f"输出特征形状: {output2.shape}")
    assert output2.shape == target_feat2.shape, f"输出形状错误: {output2.shape}"
    print("✓ 输出形状正确。")

    # 测试用例 3: 极端尺寸
    print("\n[测试3] 极端尺寸 (target_c=1024, p2_c=256, 目标尺寸4x4, P2尺寸32x32)")
    target_c3 = 1024
    p2_c3 = 256
    target_h3, target_w3 = 4, 4
    p2_h3, p2_w3 = 32, 32

    target_feat3 = torch.randn(batch, target_c3, target_h3, target_w3)
    p2_feat3 = torch.randn(batch, p2_c3, p2_h3, p2_w3)

    model3 = WeightedP2Fusion(target_channels=target_c3, p2_channels=p2_c3)
    output3 = model3(target_feat3, p2_feat3)
    print(f"目标特征形状: {target_feat3.shape}")
    print(f"P2特征形状: {p2_feat3.shape}")
    print(f"输出特征形状: {output3.shape}")
    assert output3.shape == target_feat3.shape
    print("✓ 输出形状正确。")

    # 测试用例 4: 检查融合权重的影响
    print("\n[测试4] 融合权重影响 (手动设置 alpha 观察输出变化)")
    target_feat4 = torch.ones(batch, target_c, target_h, target_w)   # 全1
    p2_feat4 = torch.zeros(batch, p2_c, p2_h, p2_w)                  # 全0

    model4 = WeightedP2Fusion(target_channels=target_c, p2_channels=p2_c)
    # 强制设置 alpha 为不同值
    with torch.no_grad():
        model4.alpha.fill_(10.0)   # sigmoid ≈ 1.0
    out_alpha1 = model4(target_feat4, p2_feat4)
    with torch.no_grad():
        model4.alpha.fill_(-10.0)  # sigmoid ≈ 0.0
    out_alpha0 = model4(target_feat4, p2_feat4)

    # 当 alpha -> 1 时，输出应接近 p2 下采样（全0），即接近 0
    # 当 alpha -> 0 时，输出应接近 target（全1）
    print(f"alpha=10 (sigmoid≈1.0) 时输出均值: {out_alpha1.mean().item():.4f} (期望接近0)")
    print(f"alpha=-10 (sigmoid≈0.0) 时输出均值: {out_alpha0.mean().item():.4f} (期望接近1)")
    assert out_alpha1.mean() < 0.1, "alpha 大时输出未接近0！"
    assert out_alpha0.mean() > 0.9, "alpha 小时输出未接近1！"
    print("✓ 权重控制正确。")

    print("\n" + "=" * 50)
    print("所有测试通过！WeightedP2Fusion 模块工作正常。")
    print("=" * 50)

if __name__ == "__main__":
    test_weighted_p2_fusion()