
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer

class CustomDetectionLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.attn_modules = [m for m in model.modules() if hasattr(m, 'get_l1_loss')]

    def __call__(self, preds, batch):
        total_loss, loss_items = super().__call__(preds, batch)
        aux_loss = sum(m.get_l1_loss() for m in self.attn_modules)
        total_loss += aux_loss
        return total_loss, loss_items

class CustomDetectionModel(DetectionModel):
    def init_criterion(self):
        return CustomDetectionLoss(self)

class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        model = CustomDetectionModel(cfg, ch=3, nc=self.data['nc'], verbose=True)
        if weights:
            model.load(weights)
        return model

# 3. 定义你的自定义训练器，重写 get_model 方法
class SmartAreaAttentionTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        # 使用自定义模型类构建模型
        model = CustomDetectionModel(cfg, ch=3, nc=self.data['nc'], verbose=True)
        if weights:
            model.load(weights)
        return model

