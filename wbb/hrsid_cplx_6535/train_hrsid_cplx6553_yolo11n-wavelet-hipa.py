
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

# ========== 热重启配置（修正版） ==========

# ---- 第一次重启（150 epoch） ----
restart1_epoch = 150
restart1_lr = 0.01             # 拉回初始学习率，制造强冲击
restart1_hold = 10             # 保持 10 个 epoch，让模型在0.01探索

# ---- 第二次重启（250 epoch） ----
restart2_epoch = 250
restart2_lr = 0.005            # 适当降低，避免后期震荡过大
restart2_hold = 10             # 保持 10 个 epoch，给最后一次充分探索

# ========== 回调函数 ==========
def on_train_epoch_start(trainer):
    if trainer.epoch == restart1_epoch:
        print(f"🔥 Epoch {restart1_epoch}: 第一次热重启, LR = {restart1_lr}, 持续 {restart1_hold} epochs")
    if trainer.epoch == restart2_epoch:
        print(f"🔥 Epoch {restart2_epoch}: 第二次热重启, LR = {restart2_lr}, 持续 {restart2_hold} epochs")

def on_train_batch_start(trainer):
    epoch = trainer.epoch
    # 第一次重启期：150 ~ 159
    if restart1_epoch <= epoch < restart1_epoch + restart1_hold:
        for pg in trainer.optimizer.param_groups:
            pg['lr'] = restart1_lr
    # 第二次重启期：250 ~ 259
    elif restart2_epoch <= epoch < restart2_epoch + restart2_hold:
        for pg in trainer.optimizer.param_groups:
            pg['lr'] = restart2_lr
    # 其余时间：让余弦退火调度器正常工作



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

    # 注册回调
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