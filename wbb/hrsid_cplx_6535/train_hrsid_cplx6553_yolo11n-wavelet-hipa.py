
from ultralytics import YOLO
import os
import torch



MODEL_NAME = "yolo11n-wavelet-hipa.yaml"
DATASET_PATH = "../cfg/hrsid_cplx_6535.yaml"


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

# ========== 热重启 + 平坦区域恒定学习率 配置 ==========
restart_epoch = 150          # 热重启点
restart_lr = 0.005           # 重启后的学习率（可调）
hold_epochs = 10             # 重启后保持恒定学习率的轮数
restart_end = restart_epoch + hold_epochs   # 160

constant_start = 250         # 平坦区域恒定学习率开始轮次
constant_lr = 0.001          # 恒定学习率值（可调，通常为初始 lr0 的 1/5 ~ 1/10）

# ========== 回调函数 ==========
def on_train_epoch_start(trainer):
    """在 epoch 开始时打印关键信息"""
    if trainer.epoch == restart_epoch:
        print(f"🔥 Epoch {restart_epoch}: Hot restart, LR set to {restart_lr}, "
              f"will hold for {hold_epochs} epochs")
    if trainer.epoch == constant_start:
        print(f"🔥 Epoch {constant_start}: Switched to constant LR = {constant_lr} "
              f"for flat region exploration until end of training")

def on_train_batch_start(trainer):
    """每个 batch 前强制覆盖学习率，保证最高优先级"""
    epoch = trainer.epoch
    # 热重启保持期：epoch 150 ~ 159
    if restart_epoch <= epoch < restart_end:
        for pg in trainer.optimizer.param_groups:
            pg['lr'] = restart_lr
    # 平坦区域恒定学习率：epoch 250 ~ 299
    elif epoch >= constant_start:
        for pg in trainer.optimizer.param_groups:
            pg['lr'] = constant_lr
    # 其他时间段：不做干预，让余弦退火调度器正常工作

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

    # ========== 注册回调 ==========
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_batch_start", on_train_batch_start)

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
            optimizer="SGD",  # 改用 SGD
            lr0=0.01,  # 初始学习率 0.01
            lrf=0.01,  # 最终学习率 = 0.01 * 0.01 = 0.0001
            momentum=0.937,  # SGD 动量
            weight_decay=0.0005,  # 权重衰减
            cos_lr=True,  # 余弦退火
            warmup_epochs=3.0,


            patience=0,  # 早停耐心值
            save=True,
            exist_ok=True,  # 覆盖现有训练结果
            verbose=True,
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