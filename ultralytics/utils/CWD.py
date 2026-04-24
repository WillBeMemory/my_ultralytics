# ============================================
# File: ultralytics/utils/distillation_loss.py
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss


class DistillationLoss(v8DetectionLoss):
    def __init__(self, model, distill_weight=0.25, T=2.0):
        super().__init__(model)
        self.distill_weight = distill_weight
        self.T = T

    def __call__(self, preds, batch):
        student_preds, teacher_preds = preds

        # 1. 处理学生预测：如果模型输出是 End2End 结构（如 YOLOv10），则取 one2many 分支计算损失；
        #    否则直接使用原输出（标准 v8/v11 格式）
        if isinstance(student_preds, dict) and 'one2many' in student_preds:
            loss_preds = student_preds['one2many']
        elif isinstance(student_preds, tuple) and isinstance(student_preds[1], dict):
            loss_preds = student_preds[1].get('one2many', student_preds[1])
        else:
            loss_preds = student_preds

        loss, loss_items = super().__call__(loss_preds, batch)

        # 2. 蒸馏损失：从学生和教师输出中，直接取出分类得分 (scores) 进行 KL 散度计算
        d_loss = torch.tensor(0.0, device=loss.device)

        # 获取学生得分
        if isinstance(student_preds, dict) and 'one2many' in student_preds:
            s_scores = student_preds['one2many']['scores']
        elif isinstance(student_preds, tuple) and isinstance(student_preds[1], dict):
            s_scores = student_preds[1].get('scores', student_preds[0])
        else:
            s_scores = student_preds[0] if isinstance(student_preds, (tuple, list)) else student_preds

        # 获取教师得分
        if isinstance(teacher_preds, dict) and 'one2many' in teacher_preds:
            t_scores = teacher_preds['one2many']['scores']
        elif isinstance(teacher_preds, tuple) and isinstance(teacher_preds[1], dict):
            t_scores = teacher_preds[1].get('scores', teacher_preds[0])
        else:
            t_scores = teacher_preds[0] if isinstance(teacher_preds, (tuple, list)) else teacher_preds

        if s_scores is not None and t_scores is not None and s_scores.shape == t_scores.shape:
            d_loss = F.kl_div(
                F.log_softmax(s_scores / self.T, dim=1),
                F.softmax(t_scores / self.T, dim=1),
                reduction='batchmean'
            ) * (self.T ** 2)

        total_loss = (1 - self.distill_weight) * loss + self.distill_weight * d_loss
        loss_items = torch.cat([loss_items, d_loss.detach().view(1)])
        return total_loss, loss_items