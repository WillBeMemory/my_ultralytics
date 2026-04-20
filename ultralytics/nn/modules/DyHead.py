import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv  # 确保 Conv 模块已导入

_all_=['DyHead']
class DyHead(nn.Module):
    """DyHead 模块，依次应用尺度、空间、任务感知注意力"""
    def __init__(self, in_channels, num_blocks=2, reduction=1/4):
        super().__init__()
        self.num_blocks = num_blocks
        self.in_channels = in_channels

        # 尺度感知注意力 (Scale-aware attention)
        self.scale_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )

        # 空间感知注意力 (Spatial-aware attention)
        self.spatial_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.spatial_norm = nn.BatchNorm2d(in_channels)

        # 任务感知注意力 (Task-aware attention)
        # 这里使用 Dynamic ReLU 的一个简化版，生成通道注意力权重
        reduced_channels = max(16, int(in_channels * reduction))
        self.task_mlp = nn.Sequential(
            nn.Linear(in_channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels * 2),
        )
        self.task_norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        """
        x: 输入特征列表 [P3, P4, P5] 或特征张量。
           如果是列表，假设形状为 [(B,C,H,W), (B,C,2H,2W), (B,C,4H,4W)] 或类似。
        """
        # 如果输入是列表，则拼接不同层级的特征
        if isinstance(x, list):
            num_levels = len(x)
            # 获取最小尺寸作为基准
            target_h, target_w = x[0].shape[2:]
            # 将所有层级上采样或下采样到同一尺寸 (例如都缩放到 P3 的大小)
            feats = []
            for i, feat in enumerate(x):
                feat = F.interpolate(feat, size=(target_h, target_w), mode='bilinear', align_corners=False)
                feats.append(feat)
            feat = torch.stack(feats, dim=0)  # [L, B, C, H, W]
        else:
            feat = x.unsqueeze(0)  # 假设是单层输入，增加层级维度
            num_levels = 1

        L, B, C, H, W = feat.shape

        # 1. 尺度感知 (Scale-aware attention)
        # 在层级维度上计算注意力，共享于空间
        scale_weights = []
        for i in range(L):
            scale_weight = self.scale_attn(feat[i])  # (B, C, 1, 1)
            scale_weights.append(scale_weight)
        scale_weights = torch.stack(scale_weights, dim=0)  # (L, B, C, 1, 1)
        feat = feat * scale_weights

        # 2. 空间感知 (Spatial-aware attention)
        # 逐层应用空间感知卷积
        for i in range(L):
            feat_i = feat[i]  # (B, C, H, W)
            # 使用分组卷积提取空间上下文
            spatial_out = self.spatial_conv(feat_i)
            spatial_out = self.spatial_norm(spatial_out)
            # 与原始特征相乘（残差风格）
            feat[i] = feat_i * spatial_out

        # 3. 任务感知 (Task-aware attention)
        # 在通道维度上计算动态权重
        # 首先将特征在空间维度上压缩
        feat_for_task = feat.mean(dim=[-2, -1])  # (L, B, C)
        # 通过 MLP 生成注意力参数
        attn_params = self.task_mlp(feat_for_task)  # (L, B, 2C)
        # 拆分为两个部分：一个用于门控，一个用于偏置
        gating, bias = torch.chunk(attn_params, 2, dim=-1)  # (L, B, C), (L, B, C)
        # 应用门控和偏置，并进行归一化
        feat = feat * torch.sigmoid(gating).view(L, B, C, 1, 1) + bias.view(L, B, C, 1, 1)
        feat = self.task_norm(feat.permute(1, 0, 2, 3, 4).contiguous()).permute(1, 0, 2, 3, 4).contiguous()

        # 如果输入是列表，将处理后的特征拆回原列表形式
        if isinstance(x, list):
            outputs = []
            for i in range(L):
                out = F.interpolate(feat[i], size=x[i].shape[2:], mode='bilinear', align_corners=False)
                outputs.append(out)
            return outputs
        else:
            return feat.squeeze(0)  # 移除层级维度