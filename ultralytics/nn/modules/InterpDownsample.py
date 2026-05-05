import torch
import torch.nn as nn
import torch.nn.functional as F

class InterpDownsample(nn.Module):
    """
    先插值到目标尺寸，再通过深度可分离卷积精炼，最后用 1×1 卷积调整通道数。
    参数：
        c1   : 输入通道数（框架自动传入）
        c2   : 输出通道数
        size : 目标正方形边长，如 128 表示 (128, 128)
        k    : 精炼卷积核大小，默认 3
    """
    def __init__(self, c1, c2, size, k=3):
        super().__init__()
        self.target_size = (size, size)
        # 深度可分离卷积精炼（保持 c1 通道）
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c1, k, stride=1, padding=k // 2, groups=c1, bias=False),
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True)
        )
        # 1×1 通道变换 (c1 -> c2)
        self.proj = nn.Conv2d(c1, c2, 1, bias=False)

    def forward(self, x):
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        x = self.conv(x)
        return self.proj(x)
# 测试代码
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}\n")

    bs, c = 2, 64
    x = torch.randn(bs, c, 160, 160).to(device)

    # 模拟框架调用：传入 c1=64, c2=128(占位，实际忽略), size=128
    model = InterpDownsample(c1=64, c2=128, size=128).to(device)
    y = model(x)
    expected = (bs, c, 128, 128)
    print(f"输入: {x.shape} -> 输出: {y.shape} (期望 {expected})")
    assert y.shape == expected, f"形状不匹配！期望 {expected}，实际 {y.shape}"
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试 size=64
    model2 = InterpDownsample(c1=64, c2=128, size=64).to(device)
    y2 = model2(x)
    expected2 = (bs, c, 64, 64)
    print(f"输入: {x.shape} -> 输出: {y2.shape} (期望 {expected2})")
    assert y2.shape == expected2, f"形状不匹配！"
    print(f"参数量: {sum(p.numel() for p in model2.parameters()):,}")
    print("\n✅ 测试通过")