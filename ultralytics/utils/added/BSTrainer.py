from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.AddModules import BackgroundSuppression


class BSTrainer(DetectionTrainer):
    """
    在每个 batch 训练前将 targets 注入模型中的所有 BackgroundSuppression 模块，
    训练结束后清除，避免影响后续验证。
    """
    def train_batch(self):
        # 获取当前 batch 的 targets（Ultralytics 标准格式：List[Instances]）
        targets = self.batch.get('instances', None)

        # 注入到所有 BackgroundSuppression 模块
        for module in self.model.modules():
            if isinstance(module, BackgroundSuppression):
                module.current_targets = targets

        # 正常训练
        super().train_batch()

        # 清除注入（避免验证时意外使用）
        for module in self.model.modules():
            if isinstance(module, BackgroundSuppression):
                module.current_targets = None