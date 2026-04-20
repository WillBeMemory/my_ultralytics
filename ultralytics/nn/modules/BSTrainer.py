import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import BackgroundSuppression
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer


# 辅助函数：生成背景掩码（保持不变）
def generate_background_mask(bboxes, feat_h, feat_w, img_h, img_w, device):
    mask = torch.ones(1, feat_h, feat_w, device=device)
    for box in bboxes:
        x1, y1, x2, y2 = box.int().tolist()
        x1 = max(0, int(x1 * feat_w / img_w))
        x2 = min(feat_w, int(x2 * feat_w / img_w))
        y1 = max(0, int(y1 * feat_h / img_h))
        y2 = min(feat_h, int(y2 * feat_h / img_h))
        if x1 < x2 and y1 < y2:
            mask[0, y1:y2, x1:x2] = 0
    return mask

class CustomDetectionLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.bg_modules = [m for m in model.modules() if isinstance(m, BackgroundSuppression)]
        self.bce_logits = nn.BCEWithLogitsLoss()
        # 背景监督损失权重（可根据需要调整）
        self.bg_weight = 1.0
        # 稀疏正则化权重（可根据需要调整）
        self.sparse_weight = 0.01

    def __call__(self, preds, batch):
        total_loss, loss_items = super().__call__(preds, batch)

        if not self.bg_modules:
            return total_loss, loss_items

        img = batch['img']
        img_h, img_w = img.shape[2], img.shape[3]
        bboxes = batch['bboxes']
        batch_idx = batch['batch_idx']

        bg_loss = 0.0
        sparse_loss = 0.0

        for module in self.bg_modules:
            # 1. 背景监督损失（使用 BCEWithLogitsLoss）
            bg_logits = module.last_bg_logits
            if bg_logits is not None:
                B, _, Hf, Wf = bg_logits.shape
                mask = torch.zeros(B, 1, Hf, Wf, device=bg_logits.device)
                for i in range(B):
                    indices = (batch_idx == i).nonzero(as_tuple=True)[0]
                    if len(indices) == 0:
                        mask[i] = 1.0
                        continue
                    boxes_i = bboxes[indices]
                    mask_i = generate_background_mask(boxes_i, Hf, Wf, img_h, img_w, bg_logits.device)
                    mask[i] = mask_i
                bg_loss += self.bce_logits(bg_logits, mask)

            # 2. 稀疏正则化损失（如果模块提供）
            if hasattr(module, 'get_sparsity_loss'):
                sparse_loss += module.get_sparsity_loss()

        total_loss += self.bg_weight * bg_loss
        total_loss += self.sparse_weight * sparse_loss

        return total_loss, loss_items

# 自定义模型和训练器（保持不变）
class CustomDetectionModel(DetectionModel):
    def init_criterion(self):
        return CustomDetectionLoss(self)

class BSTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        model = CustomDetectionModel(cfg, ch=3, nc=self.data['nc'], verbose=True)
        if weights:
            model.load(weights)
        return model