import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAwareEdgeEnhance(nn.Module):
    """
    通道感知边缘增强 + 串行精炼模块（保持原名）

    操作顺序：
    1. 通道对比度筛选（max - avg） → 通道权重
    2. 空间边缘增强（max - min） → 空间残差权重
    3. 应用双重注意力得到增强特征
    4. 经过一个轻量 Bottleneck（1x1降维 → 3x3深度卷积 → 1x1升维）进行特征精炼
    5. 残差连接（当输入输出通道相同时）

    YOLO 参数顺序：
    [c2, n, pool_size, ch_sharp, ch_thresh, edge_sharp, edge_thresh, use_refine, bottleneck_ratio, use_residual]
    n: 占位参数（无实际重复）
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        pool_size: int = 3,
        ch_sharp: float = 5.0,
        ch_thresh: float = 0.5,
        edge_sharp: float = 5.0,
        edge_thresh: float = 0.5,
        use_refine: bool = True,
        bottleneck_ratio: float = 0.5,
        use_residual: bool = True,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.ch_sharp = nn.Parameter(torch.tensor(ch_sharp), requires_grad=False)
        self.ch_thresh = nn.Parameter(torch.tensor(ch_thresh), requires_grad=False)
        self.edge_sharp = nn.Parameter(torch.tensor(edge_sharp), requires_grad=False)
        self.edge_thresh = nn.Parameter(torch.tensor(edge_thresh), requires_grad=False)
        self.use_refine = use_refine
        self.use_residual = use_residual

        # ---------- 精炼分支 ----------
        if use_refine:
            mid_channels = int(c1 * bottleneck_ratio)
            self.refine = nn.Sequential(
                nn.Conv2d(c1, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=mid_channels, bias=False),  # 深度卷积
                nn.BatchNorm2d(mid_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(mid_channels, c2, 1, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(inplace=True),
            )
        else:
            self.refine = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

    def _compute_enhancement(self, x):
        """计算增强特征图（不改变通道数）"""
        B, C, H, W = x.shape
        pad = self.pool_size // 2

        x_abs = x.abs()

        # 通道权重：max - avg
        max_ch = F.adaptive_max_pool2d(x_abs, 1)
        avg_ch = F.adaptive_avg_pool2d(x_abs, 1)
        diff_ch = max_ch - avg_ch
        ch_weight = torch.sigmoid(self.ch_sharp * (diff_ch - self.ch_thresh))

        # 空间边缘权重：max - min
        x_spatial = x_abs.mean(dim=1, keepdim=True)
        max_s = F.max_pool2d(x_spatial, self.pool_size, stride=1, padding=pad)
        min_s = -F.max_pool2d(-x_spatial, self.pool_size, stride=1, padding=pad)
        edge = max_s - min_s
        edge_weight = torch.sigmoid(self.edge_sharp * (edge - self.edge_thresh))

        enhanced = x * ch_weight
        enhanced = enhanced * (1.0 + edge_weight)
        return enhanced

    def forward(self, x):
        enhanced = self._compute_enhancement(x)       # (B, C1, H, W)
        out = self.refine(enhanced)                   # 精炼 + 通道变换 -> (B, C2, H, W)
        if self.use_residual and x.shape[1] == out.shape[1]:
            out = out + x
        return out