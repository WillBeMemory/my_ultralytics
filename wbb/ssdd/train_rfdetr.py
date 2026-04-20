"""
RF-DETR 完整训练脚本 (支持YOLO格式数据集)
适用于目标检测任务的微调训练
"""

import os
import yaml
import torch
from rfdetr import RFDETRBase  # 可根据需要更换为 RFDETRNano/Large等

# -------------------- 配置参数 --------------------
# 你可以在这里修改所有关键配置
DATASET_DIR = "./datasets/your_dataset"  # 你的数据集根目录路径
MODEL_TYPE = "RFDETRBase"  # 可选: RFDETRNano, RFDETRBase, RFDETRLarge, RFDETR2XLarge
EPOCHS = 100
BATCH_SIZE = 8  # 根据GPU显存调整
GRAD_ACCUM_STEPS = 2  # 梯度累积步数，用于模拟更大batch
LEARNING_RATE = 1e-4
OUTPUT_DIR = "./runs/rfdetr_train"  # 训练输出目录
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42  # 随机种子，设为None则每次不同


# 如果有多张GPU，可以指定使用哪些
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 取消注释以指定GPU


# -------------------- 辅助函数 --------------------
def check_yolo_dataset(dataset_path):
    """
    检查YOLO格式数据集的完整性
    YOLO格式要求:
    - 根目录下有 data.yaml
    - 存在 train/images/ 和 train/labels/ 目录
    """
    print("\n=== 检查数据集格式 (YOLO) ===")

    # 检查 data.yaml
    yaml_path = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"未找到 data.yaml: {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)

    print(f"类别数量 (nc): {data_cfg.get('nc', '未指定')}")
    print(f"类别名称: {data_cfg.get('names', [])}")

    # 检查训练集目录结构
    train_img_dir = os.path.join(dataset_path, "train", "images")
    train_label_dir = os.path.join(dataset_path, "train", "labels")

    if not os.path.exists(train_img_dir):
        raise FileNotFoundError(f"训练集图片目录不存在: {train_img_dir}")
    if not os.path.exists(train_label_dir):
        raise FileNotFoundError(f"训练集标签目录不存在: {train_label_dir}")

    # 统计图片数量
    images = [f for f in os.listdir(train_img_dir)
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    print(f"训练集图片数量: {len(images)}")

    # 可选: 检查验证集
    val_img_dir = os.path.join(dataset_path, "valid", "images")
    if os.path.exists(val_img_dir):
        val_images = [f for f in os.listdir(val_img_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"验证集图片数量: {len(val_images)}")

    print("数据集检查通过 ✓\n")
    return data_cfg


# -------------------- 主训练函数 --------------------
def train_rfdetr():
    """训练RF-DETR模型的主函数"""

    # 1. 检查数据集
    check_yolo_dataset(DATASET_DIR)

    # 2. 设置随机种子（可选）
    if SEED is not None:
        import random
        import numpy as np
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
        print(f"随机种子已固定为: {SEED}")

    # 3. 初始化模型
    print(f"正在加载 {MODEL_TYPE} 模型...")
    model = RFDETRBase()  # 首次运行会自动下载预训练权重

    # 4. 训练配置
    print("\n=== 开始训练 ===")
    print(f"设备: {DEVICE}")
    print(f"数据集: {DATASET_DIR}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"梯度累积步数: {GRAD_ACCUM_STEPS}")
    print(f"有效批次大小: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"训练轮数: {EPOCHS}")
    print(f"输出目录: {OUTPUT_DIR}")

    try:
        model.train(
            dataset_dir=DATASET_DIR,  # 数据集根目录
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            grad_accum_steps=GRAD_ACCUM_STEPS,  # 梯度累积[citation:1]
            lr=LEARNING_RATE,
            output_dir=OUTPUT_DIR,
            device=DEVICE,
            # 可选: 冻结主干网络的前几层进行迁移学习
            # freeze_backbone_layers=2,
            # 数据增强参数（使用默认值即可）
            # mosaic=1.0,
            # mixup=0.1,
        )

        print("\n✅ 训练完成！")
        print(f"模型已保存至: {OUTPUT_DIR}")

        # 5. 训练完成后，查看最佳模型路径
        best_model_path = os.path.join(OUTPUT_DIR, "checkpoint_best_total.pth")
        if os.path.exists(best_model_path):
            print(f"最佳模型: {best_model_path}")
            print("该文件仅包含模型权重，可用于推理或导出[citation:1]")

        return model

    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        return None


# -------------------- 验证测试函数 --------------------
def validate_model(model_path=None):
    """
    使用训练好的模型进行验证测试
    """
    if model_path is None:
        model_path = os.path.join(OUTPUT_DIR, "checkpoint_best_total.pth")

    if not os.path.exists(model_path):
        print(f"未找到模型文件: {model_path}")
        return

    print(f"\n=== 加载模型进行验证 ===")
    model = RFDETRBase()
    model.load_state_dict(torch.load(model_path))

    # 这里可以加载验证集进行测试
    # 例如使用 model.eval() 和 model.predict() 进行推理
    print("模型加载成功，可用于推理")


# -------------------- 脚本入口 --------------------
if __name__ == "__main__":
    # 训练模型
    trained_model = train_rfdetr()

    # 如果需要立即验证，取消下面的注释
    # if trained_model:
    #     validate_model()

    print("\n脚本执行完毕。")