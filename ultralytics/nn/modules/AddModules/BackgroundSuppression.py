import torch
import torch.nn as nn
from .global_targets import get_current_targets


class BackgroundSuppression(nn.Module):
    """
    训练时根据 GT 框动态抑制背景通道。
    推理时直接返回原始特征。
    """
    def __init__(self, ch,n, feat_size=20, tau=1.0, sharpness=5.0, use_gate=True):
        super().__init__()
        self.ch = ch
        self.feat_size = feat_size
        self.tau = tau            # 得分阈值，低于 tau 视为背景通道
        self.sharpness = sharpness # sigmoid 陡峭度
        self.use_gate = use_gate

        if self.use_gate:
            # 可选轻量门控，对抑制强度进行微调
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(ch, max(1, ch // 8), 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(1, ch // 8), ch, 1, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x):
        targets = get_current_targets()
        if targets is None or not self.training:
            # 推理或未注入 targets 时，直接返回原始特征
            return x

        B, C, H, W = x.shape
        # 生成前景掩膜 (B, H, W)
        masks = self._make_masks(targets, H, W, device=x.device)  # (B, H, W)

        # 计算每个通道在前景/背景的平均激活（使用绝对值）
        x_abs = x.abs()
        fg_mask = masks.unsqueeze(1)      # (B, 1, H, W)
        bg_mask = 1.0 - fg_mask

        fg_sum = (x_abs * fg_mask).sum(dim=[2, 3])  # (B, C)
        bg_sum = (x_abs * bg_mask).sum(dim=[2, 3])
        fg_count = fg_mask.sum(dim=[2, 3]) + 1e-6   # (B, 1)
        bg_count = bg_mask.sum(dim=[2, 3]) + 1e-6

        fg_mean = fg_sum / fg_count  # (B, C)
        bg_mean = bg_sum / bg_count

        # 前景/背景比值，越小表示越偏向背景
        score = fg_mean / (bg_mean + 1e-6)  # (B, C)

        # 生成抑制权重：score < tau 时权重接近 0（背景通道被抑制）
        weight = torch.sigmoid((score - self.tau) * self.sharpness)  # (B, C)
        weight = weight.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        if self.use_gate:
            gate_weight = self.gate(x)  # (B, C, 1, 1)
            weight = weight * gate_weight

        return x * weight

    def _make_masks(self, targets, H, W, device):
        """
        将 Ultralytics 格式的 targets 转换为 (B, H, W) 的 0/1 掩膜。
        支持 targets 为 List[ultralytics.utils.instance.Instances] 或 Tensor [N,6]。
        """
        if isinstance(targets, list):
            B = len(targets)
            masks = torch.zeros(B, H, W, device=device)
            for b, inst in enumerate(targets):
                if inst is None or len(inst.bboxes) == 0:
                    continue
                # inst.bboxes.xywhn 返回归一化中心宽高 (N,4)
                for box in inst.bboxes.xywhn:
                    cx, cy, w, h = box.tolist()
                    self._draw_box(masks[b], cx, cy, w, h, H, W)
        else:
            # 假设是 Tensor [N, 6] (batch_idx, cls, cx, cy, w, h) 归一化
            B = int(targets[:, 0].max().item()) + 1
            masks = torch.zeros(B, H, W, device=device)
            for t in targets:
                b = int(t[0].item())
                if b >= B:
                    continue
                cx, cy, w, h = t[2:6].tolist()
                self._draw_box(masks[b], cx, cy, w, h, H, W)
        return masks

    def _draw_box(self, mask_single, cx, cy, w, h, H, W):
        """将归一化框绘制到单张特征图尺寸的掩膜上"""
        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W + 1)
        y2 = int((cy + h / 2) * H + 1)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        mask_single[y1:y2, x1:x2] = 1.0