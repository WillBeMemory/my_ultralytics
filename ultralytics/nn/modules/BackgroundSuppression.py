import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['BackgroundSuppression']

class BackgroundSuppression(nn.Module):
    """
    最原始的软剪枝背景抑制模块。
    使用通道注意力 + 空间注意力生成背景 logits，然后软剪枝。
    输入输出通道相同，即插即用。
    """
    def __init__(self, in_channels, out_channels=None, reduction=16, kernel_size=7):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        assert in_channels == out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 通道注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )
        # 融合层
        self.fusion = nn.Conv2d(in_channels, 1, 1)

        # 存储背景 logits 供损失函数使用（可选）
        self.last_bg_logits = None

    def forward(self, x):
        # 通道注意力
        ca = self.channel_attn(x)
        x_ca = x * ca

        # 空间注意力
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attn(sa_input)

        # 背景 logits
        feat_map = self.fusion(x_ca)
        bg_logits = feat_map * sa
        self.last_bg_logits = bg_logits.detach()

        # 软剪枝
        bg_prob = torch.sigmoid(bg_logits)
        fg_weight = 1 - bg_prob
        out = x * fg_weight
        return out