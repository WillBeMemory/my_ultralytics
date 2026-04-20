import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv  # 使用 Ultralytics 官方 Conv 模块

__all__ = ['DynamicSPPF']

class DynamicSPPF(nn.Module):
    def __init__(self, c1, c2, k=5, n=3, shortcut=False):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)
        self.n = n
        self.shortcut = shortcut and c1 == c2
        # 可学习权重，采用 Softmax 归一化
        self.logits = nn.Parameter(torch.zeros(n + 1))

    def forward(self, x):
        y = [self.cv1(x)]
        for _ in range(self.n):
            y.append(self.m(y[-1]))
        # Softmax 归一化
        w = F.softmax(self.logits, dim=0)
        weighted_y = [w[i] * yi for i, yi in enumerate(y)]
        out = torch.cat(weighted_y, dim=1)
        out = self.cv2(out)
        if self.shortcut:
            out = out + x
        return out


# ==================== 测试代码 ====================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建模块，输入通道64，输出通道64
    model = DynamicSPPF(64, 64).to(device)

    # 随机输入 (batch=2, channels=64, height=32, width=32)
    x = torch.randn(2, 64, 32, 32).to(device)
    out = model(x)

    print(f"Input shape : {x.shape}")
    print(f"Output shape: {out.shape}")

    # 简单梯度测试
    loss = out.mean()
    loss.backward()
    print("Backward pass completed.")

    # 检查可学习权重是否有梯度
    if model.weights.grad is not None:
        print("Weights have gradients, training works.")
    else:
        print("Weights have no gradients (unexpected).")

    # 打印初始权重
    print(f"Initial weights: {model.weights.data.cpu().numpy()}")