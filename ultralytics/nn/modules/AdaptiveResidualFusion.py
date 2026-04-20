import torch
import torch.nn as nn
import torch.nn.functional as F

_all_ = ['AdaptiveResidualFusion']


class AdaptiveResidualFusion(nn.Module):
    """
    自适应残差融合模块：将下采样后的 P2 特征与目标特征进行残差融合，
    融合权重由输入特征动态生成（空间-通道自适应）。

    参数：
        target_channels: 目标特征（如 P3）的通道数
        p2_channels: P2 特征的通道数（原始，未压缩）
        compress_channels: 压缩后的中间通道数（可选，默认 target_channels//4）
    """

    def __init__(self, target_channels, p2_channels, compress_channels=None):
        super().__init__()
        if compress_channels is None:
            compress_channels = max(target_channels // 4, 16)

        # 1. 压缩 P2 通道（减少后续计算量）
        self.compress = nn.Sequential(
            nn.Conv2d(p2_channels, compress_channels, 1, bias=False),
            nn.BatchNorm2d(compress_channels),
            nn.SiLU(inplace=True)
        )

        # 2. 将压缩后的 P2 下采样到目标尺寸，并升维到 target_channels
        self.align = nn.Sequential(
            nn.Conv2d(compress_channels, target_channels, 1, bias=False),
            nn.BatchNorm2d(target_channels),
            nn.SiLU(inplace=True)
        )

        # 3. 生成自适应权重的小型网络
        self.weight_net = nn.Sequential(
            nn.Conv2d(target_channels * 2, target_channels // 4, kernel_size=3, padding=2, dilation=2),  # 空洞卷积
            nn.BatchNorm2d(target_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(target_channels // 4, target_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        输入 x 是一个列表 [target, p2]，分别对应高层特征和原始 P2 特征。
        """
        target, p2 = x  # 解包
        # 1. 压缩并下采样 P2
        x_comp = self.compress(p2)
        x_down = F.interpolate(x_comp, size=target.shape[2:], mode='bilinear', align_corners=False)
        x_aligned = self.align(x_down)

        # 2. 生成自适应权重
        concat = torch.cat([target, x_aligned], dim=1)
        weight = self.weight_net(concat)

        # 3. 残差输出
        return target + weight * x_aligned

if __name__ == "__main__":
    model = AdaptiveResidualFusion(target_channels=256, p2_channels=128, compress_channels=64)
    target = torch.randn(2, 256, 32, 32)
    p2 = torch.randn(2, 128, 64, 64)  # 假设 P2 分辨率是 target 的 2 倍
    out = model(target, p2)
    print(out.shape)  # 应输出 (2,256,32,32)
    loss = out.sum()
    loss.backward()
    print("梯度反向传播正常")