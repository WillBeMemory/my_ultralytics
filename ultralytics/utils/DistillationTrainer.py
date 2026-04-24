# ============================================
# File: ultralytics/utils/DistillationTrainer.py
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils import DEFAULT_CFG


class DistillationModel(nn.Module):
    """包装学生和教师模型，优化器仅管理学生参数"""
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.criterion = None

    def parameters(self, recurse=True):
        return self.student.parameters(recurse=recurse)

    def named_parameters(self, prefix='', recurse=True):
        return self.student.named_parameters(prefix=prefix, recurse=recurse)

    def build_loss(self, distill_weight=0.25, T=2.0):
        self.criterion = DistillationLoss(model=self.student, distill_weight=distill_weight, T=T)

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):  # 训练模式
            return self.loss(x)
        return self.student(x, *args, **kwargs)

    def loss(self, batch, preds=None):
        if preds is not None:
            # 验证阶段：只计算原始学生损失（返回4项，蒸馏损失置零）
            return self.criterion._base_loss(preds, batch)
        else:
            # 训练阶段：执行完整蒸馏流程
            img = batch['img']
            student_preds = self.student(img)
            with torch.no_grad():
                teacher_preds = self.teacher(img)
            return self.criterion((student_preds, teacher_preds), batch)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.student, name)


class DistillationLoss(v8DetectionLoss):
    def __init__(self, model, distill_weight=0.25, T=2.0):
        super().__init__(model)
        self.distill_weight = distill_weight
        self.T = T

    def _base_loss(self, preds, batch):
        """验证阶段：计算原始检测损失，补足4项"""
        loss, loss_items = super().__call__(preds, batch)
        # 补一个零蒸馏损失项，使长度与训练时一致（4项）
        zero_distill = torch.zeros(1, device=loss.device)
        loss_items = torch.cat([loss_items, zero_distill])
        return loss, loss_items

    def __call__(self, preds, batch):
        """训练阶段：原始损失 + 蒸馏损失"""
        student_preds, teacher_preds = preds
        loss, loss_items = super().__call__(student_preds, batch)

        d_loss = torch.tensor(0.0, device=loss.device)
        s_scores = self._extract_scores(student_preds)
        t_scores = self._extract_scores(teacher_preds)

        if s_scores is not None and t_scores is not None and s_scores.shape == t_scores.shape:
            d_loss = F.kl_div(
                F.log_softmax(s_scores / self.T, dim=1),
                F.softmax(t_scores / self.T, dim=1),
                reduction='batchmean'
            ) * (self.T ** 2)

        total_loss = (1 - self.distill_weight) * loss + self.distill_weight * d_loss
        loss_items = torch.cat([loss_items, d_loss.detach().view(1)])
        return total_loss, loss_items

    @staticmethod
    def _extract_scores(preds):
        if isinstance(preds, torch.Tensor):
            return preds
        if isinstance(preds, (list, tuple)):
            for item in preds:
                if isinstance(item, torch.Tensor) and item.ndim >= 3 and item.shape[-1] > 1:
                    return item
        if isinstance(preds, dict):
            for v in preds.values():
                if isinstance(v, torch.Tensor) and v.ndim >= 3 and v.shape[-1] > 1:
                    return v
        return None


class DistillationTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        self._teacher_path = overrides.pop('teacher', None)
        self._distill_weight = overrides.pop('distill_weight', 0.25)
        self._distill_temp = overrides.pop('temperature', 2.0)
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        student = super().get_model(cfg, weights, verbose)
        if not self._teacher_path:
            raise ValueError("蒸馏训练必须指定 teacher 参数")

        teacher_yolo = YOLO(self._teacher_path)
        teacher = teacher_yolo.model
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        distilled_model = DistillationModel(student, teacher)
        return distilled_model

    def set_model_attributes(self):
        super().set_model_attributes()
        self.model.student.nc = self.model.nc
        self.model.student.names = self.model.names
        self.model.student.args = self.model.args
        self.model.build_loss(distill_weight=self._distill_weight, T=self._distill_temp)

    def get_validator(self):
        validator = super().get_validator()
        self.loss_names = ["box_loss", "cls_loss", "dfl_loss", "distill_loss"]
        return validator