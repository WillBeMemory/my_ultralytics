import torch
import numpy as np
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer


class HIPALoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        # 收集所有 HIPA 模块
        self.hipa_blocks = []
        for m in model.modules():
            if hasattr(m, 'logit_thresholds') or hasattr(m, 'logit_thresholds_low'):
                self.hipa_blocks.append(m)
        self._debug_printed = False

    def _make_target_mask(self, batch, feat_size=(20, 20)):
        """根据 batch 数据生成目标掩膜 (B, H, W)"""
        # 使用官方源码中的键名: bboxes
        gt_bboxes = batch.get('bboxes', None)

        if gt_bboxes is None or len(gt_bboxes) == 0:
            B = len(batch.get('batch_idx', [0]))
            return torch.zeros(B, *feat_size, device='cpu')

        B = len(gt_bboxes)
        mask = torch.zeros(B, *feat_size, device=gt_bboxes[0].device)
        h, w = feat_size
        for i, bboxes in enumerate(gt_bboxes):
            if bboxes.numel() == 0:
                continue
            for box in bboxes:
                # bboxes 格式: (cx, cy, w, h) 归一化坐标
                cx, cy, bw, bh = box.tolist()
                x1 = cx - bw / 2
                y1 = cy - bh / 2
                x2 = cx + bw / 2
                y2 = cy + bh / 2
                gx1 = int(x1 * w)
                gy1 = int(y1 * h)
                gx2 = max(gx1 + 1, int(x2 * w + 0.5))
                gy2 = max(gy1 + 1, int(y2 * h + 0.5))
                gx1, gy1 = max(0, gx1), max(0, gy1)
                gx2, gy2 = min(w, gx2), min(h, gy2)
                if gx2 > gx1 and gy2 > gy1:
                    mask[i, gy1:gy2, gx1:gx2] = 1.0
        return mask

    def __call__(self, preds, batch):
        # 1. 生成并注入掩膜
        target_mask = self._make_target_mask(batch, feat_size=(20, 20))
        for block in self.hipa_blocks:
            block.target_mask = target_mask

        # 2. 调试打印（仅首次）
        if not self._debug_printed:
            nonzero = target_mask.sum().item()
            print(f"[HIPALoss] target_mask non-zero pixels: {nonzero:.0f}")
            if nonzero == 0:
                print("[HIPALoss] WARNING: mask is empty. Check annotation keys.")
            self._debug_printed = True

        # 3. 调用标准 YOLO 损失
        loss, loss_items = super().__call__(preds, batch)

        # 4. 清除掩膜
        for block in self.hipa_blocks:
            block.target_mask = None

        return loss, loss_items


class CustomDetectionModel(DetectionModel):
    def init_criterion(self):
        return HIPALoss(self)


class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        model = CustomDetectionModel(cfg, ch=3, nc=self.data['nc'], verbose=True)
        if weights:
            model.load(weights)
        return model