import torch
import numpy as np
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer


class HIPALoss(v8DetectionLoss):
    def __init__(self, model, reg_weight=0.1):
        super().__init__(model)
        self.reg_weight = reg_weight
        # 收集所有 HIPABlock 实例，用于注入 target_mask
        self.hipa_blocks = [m for m in model.modules() if hasattr(m, 'logit_thresholds_low')]

    def __call__(self, preds, batch):
        # 生成目标掩膜，并注入到各个 HIPABlock
        target_mask = self._make_target_mask(batch, feat_size=(20, 20))
        for block in self.hipa_blocks:
            block.target_mask = target_mask

        # 标准损失计算（包含双阈值正则化）
        loss, loss_items = super().__call__(preds, batch)

        # 清除注入的 target_mask，防止影响后续流程
        for block in self.hipa_blocks:
            block.target_mask = None

        # 双阈值正则化（鼓励阈值保持在合理范围）
        reg_loss = 0.0
        for block in self.hipa_blocks:
            low_th = block.logit_thresholds_low
            high_th = block.logit_thresholds_high
            # 目标：下阈值向 0.3 靠拢，上阈值向 0.8 靠拢
            target_low = np.log(0.3 / (1 - 0.3))
            target_high = np.log(0.8 / (1 - 0.8))
            reg_loss += ((low_th.to(loss.dtype) - target_low) ** 2).mean()
            reg_loss += ((high_th.to(loss.dtype) - target_high) ** 2).mean()
        loss += self.reg_weight * reg_loss

        return loss, loss_items

    def _make_target_mask(self, batch, feat_size):
        """根据 batch 中的标签生成二值目标掩膜 (B, 1, H, W)"""
        gt_bboxes = batch.get('gt_bboxes')
        B = len(gt_bboxes) if gt_bboxes is not None else 1
        mask = torch.zeros(B, 1, *feat_size, device=gt_bboxes[0].device if gt_bboxes else 'cpu')
        if gt_bboxes is None:
            return mask
        h, w = feat_size
        for i, bboxes in enumerate(gt_bboxes):
            if bboxes.numel() == 0:
                continue
            for box in bboxes:
                x1, y1, x2, y2 = box.tolist()
                gx1 = int(x1 * w)
                gy1 = int(y1 * h)
                gx2 = max(gx1 + 1, int(x2 * w + 0.5))
                gy2 = max(gy1 + 1, int(y2 * h + 0.5))
                gx1, gy1 = max(0, gx1), max(0, gy1)
                gx2, gy2 = min(w, gx2), min(h, gy2)
                if gx2 > gx1 and gy2 > gy1:
                    mask[i, 0, gy1:gy2, gx1:gx2] = 1.0
        return mask


class CustomDetectionModel(DetectionModel):
    def init_criterion(self):
        return HIPALoss(self, reg_weight=0.1)


class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        model = CustomDetectionModel(cfg, ch=3, nc=self.data['nc'], verbose=True)
        if weights:
            model.load(weights)
        return model