import torch
import torch.nn as nn
import math


class WIoULoss(nn.Module):
    """Wise-IoU Loss with dynamic non-monotonic focusing mechanism.
    
    Reference: https://arxiv.org/abs/2301.10051
    Official implementation: https://github.com/Instinct323/Wise-IoU
    """
    momentum = 0.01
    alpha = 1.7
    delta = 2.7

    def __init__(self, monotonous=False, eps=1e-7):
        super(WIoULoss, self).__init__()
        self.monotonous = monotonous
        self.eps = eps
        self.register_buffer('iou_running_mean', torch.tensor(1.0))

    def forward(self, pred_boxes, target_boxes):
        # 将中心点+宽高格式(xywh)转换为左上角+右下角格式(x1y1x2y2)
        b1_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        b1_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        b1_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        b1_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
        b2_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        b2_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        b2_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        b2_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

        # 计算交集面积
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        # 计算并集面积
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area + self.eps

        # IoU
        iou = inter_area / union_area
        loss_iou = 1 - iou

        # 计算最小外接框的对角线长度平方
        enclose_x1 = torch.min(b1_x1, b2_x1)
        enclose_y1 = torch.min(b1_y1, b2_y1)
        enclose_x2 = torch.max(b1_x2, b2_x2)
        enclose_y2 = torch.max(b1_y2, b2_y2)
        cw = enclose_x2 - enclose_x1
        ch = enclose_y2 - enclose_y1
        l2_box = cw ** 2 + ch ** 2 + self.eps

        # 中心点距离平方
        center_dist = ((pred_boxes[:, :2] - target_boxes[:, :2]) ** 2).sum(dim=1)

        # WIoU v1: 距离注意力 R_WIoU
        r_wiou = torch.exp(center_dist / l2_box.detach())
        loss_wiou_v1 = r_wiou * loss_iou

        # 更新运行均值
        if self.training:
            with torch.no_grad():
                self.iou_running_mean.mul_(1 - self.momentum)
                self.iou_running_mean.add_(self.momentum * loss_iou.detach().mean())

        # WIoU v3: 动态非单调聚焦
        if not self.monotonous:
            with torch.no_grad():
                # beta = L_IoU / L_IoU_mean (离群度)
                beta = loss_iou.detach() / (self.iou_running_mean + self.eps)
                # r = beta / (delta * alpha^(beta - delta))
                r_focus = beta / (self.delta * self.alpha ** (beta - self.delta))
            loss = r_focus * loss_wiou_v1
        else:
            # WIoU v2: 单调聚焦
            with torch.no_grad():
                beta = loss_iou.detach() / (self.iou_running_mean + self.eps)
            loss = beta.sqrt() * loss_wiou_v1

        return loss.mean()