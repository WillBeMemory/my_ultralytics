import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.conv import DWConv


class SlimDetect(Detect):
    """
    轻量化检测头：接收 P2, P3, P4 特征图，综合使用轻量分支与降采样重映射降低计算量。

    内部硬编码策略：
        - 所有尺度的分类/回归分支均改为一个深度可分离卷积 + 1x1 卷积，参数量大幅减少。
        - 对最高分辨率 P2 进行 2 倍降采样 → 预测 → 上采样恢复，以避免在高分辨率图上大量计算。
    """

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        # ch 为输入通道列表，形如 [c2, c3, c4]
        super().__init__(nc, reg_max, end2end, ch)  # 复用 dfl、bias_init 等

        # ----- 重构轻量化分支 -----
        # 回归分支（box head）
        c2 = max(16, ch[0] // 4, self.reg_max * 4)  # 保持与原版相同的下限，但可减小隐藏通道
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                DWConv(x, c2, 3),  # depthwise 3x3
                nn.Conv2d(c2, 4 * self.reg_max, 1)  # 1x1 投影
            ) for x in ch
        )

        # 分类分支（cls head）
        c3 = max(ch[0], min(self.nc, 100))  # 隐藏通道也可适当压缩，这里保持默认下限
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                DWConv(x, c3, 3),
                nn.Conv2d(c3, self.nc, 1)
            ) for x in ch
        )

        # 若需要 one2one 分支，同样替换
        if end2end:
            self.one2one_cv2 = nn.ModuleList(
                nn.Sequential(DWConv(x, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
            )
            self.one2one_cv3 = nn.ModuleList(
                nn.Sequential(DWConv(x, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
            )

        # 硬编码各尺度的降采样比例：P2 使用 2 倍降采样，P3、P4 原尺度过轻量分支
        self.slim_scales = [0.5, 1.0, 1.0]

    def forward_head(self, x, box_head=None, cls_head=None, **kwargs):
        """
        前向过程：根据 slim_scales 对特征图进行可选的降采样/上采样包裹，然后送入轻量分支。
        """
        if box_head is None or cls_head is None:
            return dict()
        bs = x[0].shape[0]
        box_outs, cls_outs = [], []

        for i, feat in enumerate(x):
            H_orig, W_orig = feat.shape[-2:]
            scale = self.slim_scales[i]
            if scale < 1.0:
                # 降采样 → 预测 → 上采样恢复
                H_low = int(H_orig * scale)
                W_low = int(W_orig * scale)
                feat_low = F.interpolate(feat, size=(H_low, W_low), mode='nearest')
                box_low = box_head[i](feat_low)
                cls_low = cls_head[i](feat_low)
                box_out = F.interpolate(box_low, size=(H_orig, W_orig), mode='nearest')
                cls_out = F.interpolate(cls_low, size=(H_orig, W_orig), mode='nearest')
            else:
                box_out = box_head[i](feat)
                cls_out = cls_head[i](feat)

            box_outs.append(box_out.view(bs, 4 * self.reg_max, -1))
            cls_outs.append(cls_out.view(bs, self.nc, -1))

        boxes = torch.cat(box_outs, dim=-1)
        scores = torch.cat(cls_outs, dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)