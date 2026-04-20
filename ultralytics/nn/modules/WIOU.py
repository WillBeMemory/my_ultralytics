import torch
import torch.nn as nn
import math


class WIoULoss(nn.Module):
    def __init__(self, monotonous=False, eps=1e-7):
        super(WIoULoss, self).__init__()
        self.monotonous = monotonous
        self.eps = eps

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

        # 计算中心点距离
        center_dist = ((pred_boxes[:, :2] - target_boxes[:, :2]) ** 2).sum(dim=1)

        # 计算最小外接框的对角线长度
        enclose_x1 = torch.min(b1_x1, b2_x1)
        enclose_y1 = torch.min(b1_y1, b2_y1)
        enclose_x2 = torch.max(b1_x2, b2_x2)
        enclose_y2 = torch.max(b1_y2, b2_y2)
        enclose_diag = ((enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2) + self.eps

        # WIoU v1: 距离注意力
        wiou_term = torch.exp(center_dist / enclose_diag)
        loss_iou = 1 - iou

        # WIoU v3: 动态非单调聚焦
        if not self.monotonous:
            # 计算离群度 beta
            with torch.no_grad():
                loss_iou_mean = loss_iou.mean().detach()
                beta = loss_iou / (loss_iou_mean + self.eps)
                # 聚焦系数 r = beta / (delta * alpha^(beta - delta))
                delta = torch.mean(beta)  # 简单实现，可根据论文调整
                alpha = 1.5  # 论文中推荐值
                r = beta / (delta * alpha ** (beta - delta))
                r = torch.clamp(r, max=3.0)  # 限制最大值避免梯度爆炸
            loss = r * wiou_term * loss_iou
        else:
            loss = wiou_term * loss_iou

        return loss.mean()