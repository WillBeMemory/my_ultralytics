import math
import torch
import torch.nn as nn

class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA) 模块。

    通过自适应一维卷积快速建模通道依赖，避免降维造成的特征损失。
    参数:
        channels (int): 输入特征图的通道数。
        gamma   (int): 控制核大小的超参数，默认 2。
        b       (int): 控制核大小的超参数，默认 1。
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # 自适应计算一维卷积的核大小
        t = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = t if t % 2 else t + 1  # 保证核大小为奇数
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # 全局平均池化 → (B, C, 1, 1)
        y = x.mean(dim=(2, 3), keepdim=True)
        # 转换为序列格式: (B, C, 1) → (B, 1, C)
        y = y.squeeze(-1).transpose(-1, -2)
        # 一维卷积捕捉跨通道交互
        y = self.conv(y)                      # (B, 1, C)
        # 恢复形状并激活
        y = y.transpose(-1, -2).unsqueeze(-1) # (B, C, 1, 1)
        y = self.sigmoid(y)
        # 通道加权
        return x * y


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # 创建一个 ECA 模块，通道数 64
    channels = 64
    eca = ECA(channels).to(device)

    # 随机输入 (batch=2, channels=64, height=32, width=32)
    x = torch.randn(2, channels, 32, 32).to(device)
    out = eca(x)

    # 打印结果
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Input and output shapes match: {x.shape == out.shape}")

    # 参数量统计
    total_params = sum(p.numel() for p in eca.parameters())
    print(f"ECA total parameters: {total_params:,}")

    # 示例：ECA 一般只引入个位数参数
    # 对于 channels=64，计算出的 kernel_size 为 3 或 5，参数仅 3 或 5
    print("ECA is extremely lightweight!")
