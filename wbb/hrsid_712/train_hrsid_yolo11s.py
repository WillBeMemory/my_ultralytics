# train_hrsid_yolo11s.py — 基线 YOLO11s，7:1:2 划分 (hrsid_712.yaml)
# 由 wbb/hrsid/train_hrsid_yolo11s.py 复制，仅改 DATASET_PATH，其余协议保持一致以保证 delta 可比。
# 数据集: HRSID_YOLO_712  3923 train / 560 val / 1121 test

from ultralytics import YOLO
import os
import torch

MODEL_NAME = "yolo11s.yaml"
DATASET_PATH = "../cfg/hrsid_712.yaml"


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

    # 训练配置
    print("开始训练 船舶检测模型...")
    try:
        results = model.train(
            data=DATASET_PATH,
            epochs=300,
            imgsz=640,
            batch=16,
            workers=0,  # Windows 下设为 0 避免多进程问题
            device=device,
            optimizer="SGD",  # 改用 SGD
            lr0=0.01,  # 初始学习率 0.01
            lrf=0.01,  # 最终学习率 = 0.01 * 0.01 = 0.0001

            patience=0,  # 早停耐心值
            save=True,
            verbose=True,
        )

        print("训练完成!")
        return results

    except Exception as e:
        print(f"训练失败: {e}")
        return None


if __name__ == "__main__":
    train_model()
