# train_hrsc.py
from collections import defaultdict

from ultralytics import YOLO
import os
import torch
from ultralytics import YOLO
from ultralytics.utils.CustomLoss import CustomTrainer

MODEL_NAME = "yolo11n-dgc-moe.yaml"
DATASET_PATH = "../cfg/cplx.yaml"


from collections import defaultdict

class BranchStatsCallback:
    def __init__(self):
        self.reset()

    def reset(self):
        self.probs_sum = defaultdict(float)   # 每个模块的概率累计
        self.probs_count = defaultdict(int)   # 每个模块的 batch 计数

    def adjust_temperature_and_hardening(self,model, epoch, total_epochs=300,
                                         tau_start=5.0, tau_end=1.0,
                                         hard_epoch=200):
        """
        对所有支持 tau / hard 的模块执行温度退火和硬化。

        Args:
            model: 训练中的模型
            epoch: 当前 epoch（从 0 或 1 开始均可，内部按相对比例）
            total_epochs: 总训练轮数
            tau_start: 初始温度
            tau_end: 最终温度（硬化前）
            hard_epoch: 开始硬化的 epoch（之后 hard=True，tau 不再变化）

        Returns:
            tau, hard: 当前 epoch 使用的温度值和硬化标志
        """
        if epoch < hard_epoch:
            # 线性退火：从 tau_start 线性下降到 tau_end
            ratio = epoch / hard_epoch
            tau = tau_start - (tau_start - tau_end) * ratio
            hard = False
        else:
            tau = tau_end  # 硬化后温度不再有意义，可保留终值
            hard = True

        # 递归修改所有符合条件的子模块
        for module in model.modules():
            if hasattr(module, 'tau') and hasattr(module, 'hard'):
                module.tau = tau
                module.hard = hard

        return tau, hard

    def on_train_batch_end(self, trainer):
        """每个 batch 结束后累加各模块的 current_probs"""
        model = trainer.model.model  # 实际的 nn.Module
        for name, module in model.named_modules():
            if hasattr(module, 'current_probs') and hasattr(module, 'regularization_loss'):
                probs = module.current_probs.detach().cpu()  # 假设形状为 (3,)
                self.probs_sum[name] += probs
                self.probs_count[name] += 1

    def on_train_epoch_end(self, trainer):
        """每个 epoch 结束时更新 MoE 和 TernaryDPConv 的温度与硬化标志"""
        epoch = trainer.epoch  # 当前 epoch（从 0 开始）
        total_epochs = trainer.epochs  # 总 epoch 数（例如 300）

        # # 调用统一调度器（可根据需要调整参数）
        # tau, hard = self.adjust_temperature_and_hardening(
        #     model=trainer.model,
        #     epoch=epoch,
        #     total_epochs=total_epochs,
        #     tau_start=5.0,
        #     tau_end=1.0,
        #     hard_epoch=200
        # )

        """每个 epoch 结束后打印平均分支概率"""
        print("\n========== TernaryDPConv Branch Selection Probabilities ==========")
        for name in self.probs_sum:
            avg_probs = self.probs_sum[name] / self.probs_count[name]
            # 格式化输出：剪枝、恒等、卷积
            print(f"  {name}: prune={avg_probs[0]:.4f}, identity={avg_probs[1]:.4f}, conv={avg_probs[2]:.4f}")
        print("===================================================================\n")
        self.reset()  # 重置计数器，准备下一个 epoch

        total_samples = len(trainer.train_loader.dataset)
        for module in trainer.model.model.modules():
            if hasattr(module, 'expert_usage'):
                # expert_usage 是累计次数，除以总样本数得平均概率
                usage_prob = module.expert_usage / total_samples
                print(f"Epoch {trainer.epoch + 1} Expert usage (avg probability): {usage_prob.cpu().numpy()}")
                module.reset_usage()

    def on_train_epoch_start(self,trainer):
        for module in trainer.model.model.modules():
            if hasattr(module, 'reset_usage'):
                module.reset_usage()


def setup_environment():
    """设置训练环境"""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'


def check_dataset(config_path):
    """检查数据集配置"""
    import yaml

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("=== 数据集检查 ===")
    print(f"数据集路径: {config['path']}")
    print(f"训练集: {config['train']}")
    print(f"验证集: {config['val']}")
    print(f"类别数量: {config['nc']}")
    print(f"类别名称: {list(config['names'].values())}")

    # 检查路径是否存在
    train_path = os.path.join(config['path'], config['train'])
    val_path = os.path.join(config['path'], config['val'])

    if os.path.exists(train_path):
        train_images = [f for f in os.listdir(train_path) if f.lower().endswith(('.bmp', '.jpg', '.png', '.jpeg'))]
        print(f"训练集图像数量: {len(train_images)}")
    else:
        print(f"警告: 训练集路径不存在 - {train_path}")

    if os.path.exists(val_path):
        val_images = [f for f in os.listdir(val_path) if f.lower().endswith(('.bmp', '.jpg', '.png', '.jpeg'))]
        print(f"验证集图像数量: {len(val_images)}")
    else:
        print(f"警告: 验证集路径不存在 - {val_path}")

    print("==========================\n")


def train_model():
    """训练 hrsid 船舶检测模型"""

    setup_environment()

    # 检查数据集
    check_dataset(DATASET_PATH)

    # 检查 CUDA
    device = 'cpu'

    if torch.cuda.is_available():
        device = 0
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用 CPU 训练")

    # 加载模型 - 从配置文件开始（不使用预训练权重）
    print("加载 YOLO 模型...")
    model = YOLO(MODEL_NAME)  # 从配置文件开始

    # 创建回调实例
    stats_cb = BranchStatsCallback()

    # 注册回调
    model.add_callback('on_train_epoch_start', stats_cb.on_train_epoch_start)
    model.add_callback('on_train_batch_end', stats_cb.on_train_batch_end)
    model.add_callback('on_train_epoch_end', stats_cb.on_train_epoch_end)


    # 训练配置
    print("开始训练 船舶检测模型...")
    try:
        results = model.train(
            data=DATASET_PATH,
            epochs=300,
            imgsz=800,
            batch=16,
            workers=0,  # Windows 下设为 0 避免多进程问题
            device=device,
            lr0=0.01,  # 初始学习率
            weight_decay=0.0005,
            warmup_epochs=3.0,
            patience=0,  # 早停耐心值
            save=True,
            exist_ok=True,  # 覆盖现有训练结果
            verbose=True,
            trainer=CustomTrainer,
            # 数据增强 - 针对船舶检测优化
            # degrees=5.0,  # 较小的旋转角度（船舶方向相对固定）
            # translate=0.1,
            # scale=0.5,
            # shear=1.0,  # 较小的剪切变换
            # perspective=0.0005,  # 较小的透视变换
            # flipud=0.3,  # 较小的上下翻转概率
            # fliplr=0.5,  # 左右翻转概率
            # mosaic=1.0,
            # mixup=0.1,
        )

        print("训练完成!")
        return results

    except Exception as e:
        print(f"训练失败: {e}")
        return None


if __name__ == "__main__":
    train_model()