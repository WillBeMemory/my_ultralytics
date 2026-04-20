# train_hrsc.py

from ultralytics import YOLO
import os
import torch

MODEL_NAME = "yolo11-dwwtconv-dywtattn.yaml"
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

    # 编译模型（仅当 PyTorch >= 2.0 且 GPU 支持时）
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        print("启用 torch.compile 加速...")
        model.model = torch.compile(model.model, mode="reduce-overhead")
    else:
        print("torch.compile 不可用，使用普通模式")

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
            amp = True,  # 启用混合精度训练
            # 数据增强参数
            # hsv_h=0.0,  # SAR灰度图，色相不变
            # hsv_s=0.0,
            # hsv_v=0.2,  # 适当调整亮度对比度（模拟不同海况）
            # degrees=180.0,  # 允许任意角度旋转（舰船任意方向）
            # translate=0.1,  # 平移
            # scale=0.5,  # 缩放（模拟尺度变化）
            # shear=0.0,  # 剪切（可选，SAR图像变形少）
            # perspective=0.0,  # 透视（一般不使用）
            # flipud=0.5,  # 上下翻转（舰船方向任意，可用）
            # fliplr=0.5,  # 左右翻转
            # mosaic=1.0,  # 马赛克增强（小数据集强烈推荐）
            # mixup=0.5,  # 混合增强（小数据集推荐）
            # copy_paste=0.0,  # 复制粘贴（OBB任务可选，但需注意标注）
            # erasing=0.0,  # 随机擦除（可选）
            # crop_fraction=0.0  # 随机裁剪（不常用）
        )

        print("训练完成!")
        return results

    except Exception as e:
        print(f"训练失败: {e}")
        return None


if __name__ == "__main__":
    train_model()