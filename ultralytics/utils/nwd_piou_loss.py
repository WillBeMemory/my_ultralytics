import torch
import torch.nn as nn
import math


class NWD_PIoULoss(nn.Module):
    """
    NWD + PIoUv2 复合损失函数
    结合了NWD对小目标的敏感性和PIoUv2的非单调注意力机制
    """

    def __init__(self, nwd_weight=0.5, piou_weight=0.5, nwd_constant=12.8,
                 piou_gamma=0.5, piou_lambda=0.5, eps=1e-7):
        """
        Args:
            nwd_weight: NWD损失权重
            piou_weight: PIoU损失权重
            nwd_constant: NWD归一化常数C，根据数据集调整（小目标建议8-12）
            piou_gamma: PIoU注意力函数的gamma参数
            piou_lambda: PIoU惩罚项系数
            eps: 防止除零
        """
        super().__init__()
        self.nwd_weight = nwd_weight
        self.piou_weight = piou_weight
        self.nwd_constant = nwd_constant
        self.piouw_gamma = piou_gamma
        self.piouw_lambda = piou_lambda
        self.eps = eps

    def _nwd_loss(self, pred_bboxes, target_bboxes):
        """
        计算NWD损失
        Args:
            pred_bboxes: (N, 4) 预测框，格式 [x1, y1, x2, y2]
            target_bboxes: (N, 4) 真实框，格式 [x1, y1, x2, y2]
        Returns:
            loss: (N,) 每个样本的NWD损失 (1 - NWD)
        """
        # 转换为 [cx, cy, w, h] 格式
        pred_x = (pred_bboxes[:, 0] + pred_bboxes[:, 2]) / 2
        pred_y = (pred_bboxes[:, 1] + pred_bboxes[:, 3]) / 2
        pred_w = pred_bboxes[:, 2] - pred_bboxes[:, 0]
        pred_h = pred_bboxes[:, 3] - pred_bboxes[:, 1]

        target_x = (target_bboxes[:, 0] + target_bboxes[:, 2]) / 2
        target_y = (target_bboxes[:, 1] + target_bboxes[:, 3]) / 2
        target_w = target_bboxes[:, 2] - target_bboxes[:, 0]
        target_h = target_bboxes[:, 3] - target_bboxes[:, 1]

        # Wasserstein距离计算 [citation:1][citation:3]
        center_dist = (pred_x - target_x) ** 2 + (pred_y - target_y) ** 2
        wh_dist = ((pred_w - target_w) ** 2 + (pred_h - target_h) ** 2) / 4

        wasserstein_2 = center_dist + wh_dist + self.eps

        # 归一化得到NWD [citation:1]
        nwd = torch.exp(-torch.sqrt(wasserstein_2) / self.nwd_constant)

        # NWD损失
        return 1 - nwd

    def _piouv2_loss(self, pred_bboxes, target_bboxes):
        """
        计算PIoU v2损失
        参考PIoU论文的实现
        """
        # 基础IoU计算
        inter_x1 = torch.max(pred_bboxes[:, 0], target_bboxes[:, 0])
        inter_y1 = torch.max(pred_bboxes[:, 1], target_bboxes[:, 1])
        inter_x2 = torch.min(pred_bboxes[:, 2], target_bboxes[:, 2])
        inter_y2 = torch.min(pred_bboxes[:, 3], target_bboxes[:, 3])
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        pred_area = (pred_bboxes[:, 2] - pred_bboxes[:, 0]) * (pred_bboxes[:, 3] - pred_bboxes[:, 1])
        target_area = (target_bboxes[:, 2] - target_bboxes[:, 0]) * (target_bboxes[:, 3] - target_bboxes[:, 1])
        union_area = pred_area + target_area - inter_area + self.eps

        iou = inter_area / union_area

        # 转换为 [cx, cy, w, h] 格式用于惩罚项
        pred_cx = (pred_bboxes[:, 0] + pred_bboxes[:, 2]) / 2
        pred_cy = (pred_bboxes[:, 1] + pred_bboxes[:, 3]) / 2
        pred_w = pred_bboxes[:, 2] - pred_bboxes[:, 0]
        pred_h = pred_bboxes[:, 3] - pred_bboxes[:, 1]

        target_cx = (target_bboxes[:, 0] + target_bboxes[:, 2]) / 2
        target_cy = (target_bboxes[:, 1] + target_bboxes[:, 3]) / 2
        target_w = target_bboxes[:, 2] - target_bboxes[:, 0]
        target_h = target_bboxes[:, 3] - target_bboxes[:, 1]

        # 中心点距离惩罚项
        center_dist = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

        # 最小外接框对角线长度
        enclose_x1 = torch.min(pred_bboxes[:, 0], target_bboxes[:, 0])
        enclose_y1 = torch.min(pred_bboxes[:, 1], target_bboxes[:, 1])
        enclose_x2 = torch.max(pred_bboxes[:, 2], target_bboxes[:, 2])
        enclose_y2 = torch.max(pred_bboxes[:, 3], target_bboxes[:, 3])
        enclose_diag = ((enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2) + self.eps

        # 宽高比惩罚项（PIoU的自适应惩罚因子）
        dw1 = torch.abs(pred_cx - target_cx)
        dw2 = torch.abs(pred_w - target_w)
        dh1 = torch.abs(pred_cy - target_cy)
        dh2 = torch.abs(pred_h - target_h)

        # PIoU的惩罚因子
        P = ((dw1 + dw2) / torch.abs(target_w) + (dh1 + dh2) / torch.abs(target_h)) / 4

        # PIoU v2的非单调注意力函数
        q = torch.exp(-P)
        x = q * self.piouw_lambda

        # PIoU v2损失
        L_piou_v1 = 1 - iou - torch.exp(-P ** 2) + 1
        L_piou_v2 = 3 * x * torch.exp(-x ** 2) * L_piou_v1

        return L_piou_v2

    def forward(self, pred_bboxes, target_bboxes):
        """
        计算复合损失
        Returns:
            loss: 加权后的总损失
            nwd_part: NWD部分损失（用于监控）
            piou_part: PIoU部分损失（用于监控）
        """
        nwd_loss = self._nwd_loss(pred_bboxes, target_bboxes)
        piou_loss = self._piouv2_loss(pred_bboxes, target_bboxes)

        # 加权组合
        total_loss = self.nwd_weight * nwd_loss + self.piou_weight * piou_loss

        return total_loss, nwd_loss.mean(), piou_loss.mean()