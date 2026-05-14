from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.AddModules import BackgroundSuppression  # 根据实际路径修改
from ultralytics.utils import LOGGER
import sys

class BSTrainer(DetectionTrainer):
    """
    在每个 batch 训练前，将当前 batch 的 GT 框注入所有 BackgroundSuppression 模块，
    训练后自动清除，避免影响验证/推理。
    """
    def train_batch(self):
        sys.stderr.write("[BSTrainer] train_batch was called!\n")
        sys.stderr.flush()
        # 1. 获取当前 batch 的 targets（标准格式：List[ultralytics.utils.instance.Instances]）
        targets = self.batch.get('instances', None)

        # 2. 打印调试信息（使用 LOGGER，确保在 Ultralytics 日志系统中可见）
        if isinstance(targets, list):
            total_boxes = sum(len(inst.bboxes) for inst in targets if inst is not None)
            LOGGER.info(f"[BSTrainer] Batch: {len(targets)} images, {total_boxes} boxes")
        else:
            LOGGER.info(f"[BSTrainer] targets type: {type(targets)}, value: {targets}")

        # 3. 将 targets 注入到所有 BackgroundSuppression 模块
        for module in self.model.modules():
            if isinstance(module, BackgroundSuppression):
                module.current_targets = targets

        # 4. 执行原始训练逻辑（前向 → 损失 → 反向）
        super().train_batch()

        # 5. 清除注入，避免后续验证/推理时模块误用 GT
        for module in self.model.modules():
            if isinstance(module, BackgroundSuppression):
                module.current_targets = None