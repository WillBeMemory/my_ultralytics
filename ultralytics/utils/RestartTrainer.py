import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


class RestartTrainer(DetectionTrainer):
    """自定义Trainer：使用 CosineAnnealingWarmRestarts 替代默认的余弦退火"""

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9,
                        decay=1e-5, iterations=1e5):
        """先调用父类构建优化器，然后替换学习率调度器"""
        # 1. 用父类方法构建优化器（保留所有自动参数组、权重衰减等逻辑）
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)

        # 2. 用 CosineAnnealingWarmRestarts 替换调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50,        # 第一个周期持续 50 个 epoch
            T_mult=2,      # 每个后续周期长度翻倍（50 → 100 → 200 → ...）
            eta_min=lr * 0.01,  # 每个周期的最低学习率 = lr0 * 0.01
        )
        # 注意：YOLO 的 scheduler.step() 在每 epoch 结束时自动调用，无需手动干预

        return optimizer

