from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss

# 1. 定义你的自定义损失类，它需要继承原始的v8DetectionLoss
class CustomLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        # 收集所有需要计算辅助损失的模块
        self.aux_modules = [m for m in model.modules() if hasattr(m, 'regularization_loss')]

    def __call__(self, preds, batch):
        # 计算原始的检测损失
        total_loss, loss_items = super().__call__(preds, batch)

        # 累加所有自定义模块的辅助损失
        aux_loss_total = 0.0
        for module in self.aux_modules:
            aux_loss_total += module.regularization_loss()

        # 将辅助损失加到总损失中，返回即可
        total_loss += aux_loss_total
        return total_loss, loss_items

# 2. 定义你的自定义模型，重写 init_criterion 方法
class CustomDetectionModel(DetectionModel):
    def init_criterion(self):
        # 返回你在上一步定义的自定义损失类
        return CustomLoss(self)

# 3. 定义你的自定义训练器，重写 get_model 方法
class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        # 创建一个自定义模型类的实例
        model = CustomDetectionModel(cfg, ch=3, nc=self.data['nc'], verbose=True)
        if weights:
            model.load(weights)
        return model
