import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import DWConv, Conv
from ultralytics.nn.modules.head import Detect


class SlimDetect(Detect):
    """
    极速版 SlimDetect：利用降采样‑预测‑上采样减少高分辨率层计算量，
    同时通过合并插值、nearest 模式、可调压缩比来最小化时间开销。
    """
    def __init__(self, nc=80, reg_max=16, end2end=False, ch=(),
                 slim_indices=(), slim_scale=0.5, slim_mode="nearest",
                 c2_ratio=0.5, c3_ratio=0.5):
        super().__init__(nc, reg_max, end2end, ch)
        self.slim_indices = slim_indices
        self.slim_scale = slim_scale
        self.slim_mode = slim_mode

        # 如果压缩比 < 1，重新构建更窄的测头
        if c2_ratio < 1.0 or c3_ratio < 1.0:
            c2_ = max(16, int(ch[0] // 4 * c2_ratio), self.reg_max * 4)
            c3_ = max(ch[0], min(self.nc, 100), int(ch[0] * c3_ratio))
            self.cv2 = nn.ModuleList(
                nn.Sequential(Conv(x, c2_, 3), Conv(c2_, c2_, 3), nn.Conv2d(c2_, 4 * self.reg_max, 1))
                for x in ch
            )
            self.cv3 = nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3_, 1)),
                    nn.Sequential(DWConv(c3_, c3_, 3), Conv(c3_, c3_, 1)),
                    nn.Conv2d(c3_, self.nc, 1),
                )
                for x in ch
            )

    def forward_head(self, x, box_head=None, cls_head=None, **kwargs):
        if box_head is None or cls_head is None:
            return dict()
        bs = x[0].shape[0]
        box_outs, cls_outs = [], []

        for i in range(self.nl):
            feat = x[i]
            H, W = feat.shape[2:]
            if i in self.slim_indices:
                H_low, W_low = int(H * self.slim_scale), int(W * self.slim_scale)
                feat_low = F.interpolate(feat, size=(H_low, W_low), mode=self.slim_mode)
                box_low = box_head[i](feat_low)
                cls_low = cls_head[i](feat_low)
                # 合并插值：将 box 和 cls 在通道上拼接，一次上采样，再拆分
                combined = torch.cat([box_low, cls_low], dim=1)  # (B, 4*reg_max+nc, H_low, W_low)
                combined_up = F.interpolate(combined, size=(H, W), mode=self.slim_mode)
                reg_max4 = 4 * self.reg_max
                box_out, cls_out = combined_up.split([reg_max4, self.nc], dim=1)
            else:
                box_out = box_head[i](feat)
                cls_out = cls_head[i](feat)
            box_outs.append(box_out.view(bs, 4 * self.reg_max, -1))
            cls_outs.append(cls_out.view(bs, self.nc, -1))

        boxes = torch.cat(box_outs, dim=-1)
        scores = torch.cat(cls_outs, dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)