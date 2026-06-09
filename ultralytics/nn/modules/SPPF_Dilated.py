import torch
import torch.nn as nn
from ultralytics.nn.modules.block import Conv


class SPPF_Dilated(nn.Module):
    """
    Dilated Spatial Pyramid Pooling - Fast
    用多尺度空洞卷积分支替代原始 SPPF 的串行 maxpool
    - 并行 4 个不同 dilation 的 3×3 深度卷积 (dilation=1,3,5,7)
    - 在不显著增加参数量的前提下，大幅扩展感受野
    - 特别适用于 SAR 图像中船只与大范围背景的上下文建模
    """

    def __init__(self, c1, c2, k=5, dilations=(1, 3, 5, 7)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        # 多尺度空洞卷积分支（深度可分离，参数量可控）
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c_, c_, 3, 1, padding=d, dilation=d, groups=c_, bias=False),
                nn.BatchNorm2d(c_),
                nn.SiLU(inplace=True),
            ) for d in dilations
        ])
        # 融合 (原始 + 各分支)
        self.cv2 = Conv(c_ * (len(dilations) + 1), c2, 1, 1)

    def forward(self, x):
        y = self.cv1(x)
        feats = [y]
        for branch in self.branches:
            feats.append(branch(y))
        return self.cv2(torch.cat(feats, 1))
