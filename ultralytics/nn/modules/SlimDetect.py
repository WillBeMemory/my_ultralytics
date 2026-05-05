import torch
from torch import nn

from ultralytics.nn.modules.block import DWConv
from ultralytics.nn.modules.head import Detect


class SlimDetect(Detect):
    """
    轻量检测头，专用于 (128, 64, 32) 特征图
    内部使用深度可分离卷积降低计算量
    """

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        # 调用父类初始化，框架会自动计算 stride (5, 10, 20) 并生成 anchors
        super().__init__(nc, reg_max, end2end, ch)

        # 覆盖回归头和分类头，使用更轻量的卷积
        c2 = max(16, ch[0] // 4, self.reg_max * 4)
        c3 = max(ch[0], min(self.nc, 100))

        self.cv2 = nn.ModuleList(
            nn.Sequential(DWConv(x, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(DWConv(x, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
        )
        # 若需要 one2one 分支，此处也应同样替换，但通常训练时不需要