# val_fixed.py

from ultralytics import YOLO
import cv2
import os
import numpy as np


def validate_trained_model():
    """验证训练好的模型"""

    # 加载训练好的最佳模型
    model_path = "runs/detect/train/weights/best.pt"

    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        # 尝试查找其他可能的模型文件
        train_dir = "runs/detect/train/weights"
        if os.path.exists(train_dir):
            for file in os.listdir(train_dir):
                if file.endswith('.pt'):
                    model_path = os.path.join(train_dir, file)
                    print(f"使用模型: {model_path}")
                    break

    if not os.path.exists(model_path):
        print("错误: 找不到任何模型文件")
        return

    model = YOLO(model_path)

    # 在验证集上评估
    print("在验证集上评估模型...")
    try:
        metrics = model.val(
            data="../cfg/ssdd.yaml",
            split='val',  # 明确指定使用验证集
            verbose=True
        )

        print("\n=== 评估结果 ===")

        # 修复：正确处理数组类型的指标
        if hasattr(metrics, 'box'):
            # mAP 指标
            print(f"mAP50: {metrics.box.map50:.4f}")
            print(f"mAP50-95: {metrics.box.map:.4f}")

            # 精确率和召回率 - 取平均值或显示所有类别
            if hasattr(metrics.box, 'p'):
                precision = metrics.box.p
                if isinstance(precision, (np.ndarray, list)):
                    if len(precision) > 0:
                        print(f"平均精确率: {np.mean(precision):.4f}")
                        print(f"各类别精确率: {[f'{p:.4f}' for p in precision]}")
                    else:
                        print("精确率数据为空")
                else:
                    print(f"精确率: {precision:.4f}")

            if hasattr(metrics.box, 'r'):
                recall = metrics.box.r
                if isinstance(recall, (np.ndarray, list)):
                    if len(recall) > 0:
                        print(f"平均召回率: {np.mean(recall):.4f}")
                        print(f"各类别召回率: {[f'{r:.4f}' for r in recall]}")
                    else:
                        print("召回率数据为空")
                else:
                    print(f"召回率: {recall:.4f}")

            # 按类别显示 AP
            if hasattr(metrics.box, 'ap_class_index') and hasattr(metrics.box, 'ap50'):
                print("\n各类别 AP50:")
                for i, class_idx in enumerate(metrics.box.ap_class_index):
                    class_name = metrics.names[class_idx] if hasattr(metrics,
                                                                     'names') and class_idx in metrics.names else f'Class_{class_idx}'
                    ap50 = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0
                    print(f"  {class_name}: {ap50:.4f}")

        # 显示其他可用的指标
        print(f"\n所有可用属性: {[attr for attr in dir(metrics.box) if not attr.startswith('_')]}")

    except Exception as e:
        print(f"评估失败: {e}")
        # 尝试备选评估方法
        alternative_validation(model)


def alternative_validation(model):
    """备选验证方法"""
    print("\n尝试备选验证方法...")

    try:
        # 方法1: 使用 predict 并手动计算指标
        results = model.predict(
            source="../datasets/SSDD/images/val",
            save=False,
            conf=0.25,
            iou=0.5,
            verbose=False
        )

        # 统计预测结果
        total_detections = 0
        confident_detections = 0

        for result in results:
            if result.boxes is not None:
                total_detections += len(result.boxes)
                confident_detections += sum(result.boxes.conf > 0.5)

        print(f"总检测数: {total_detections}")
        print(f"高置信度检测数 (conf > 0.5): {confident_detections}")

    except Exception as e:
        print(f"备选验证也失败: {e}")


def visualize_predictions():
    """可视化一些预测结果"""
    model_path = "runs/detect/train/weights/best.pt"

    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return

    model = YOLO(model_path)

    # 验证集图像目录
    val_images_dir = "../datasets/SSDD/images/val"

    if not os.path.exists(val_images_dir):
        print(f"验证集目录不存在: {val_images_dir}")
        return

    # 获取前几张图像
    image_files = [f for f in os.listdir(val_images_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))][:3]

    if not image_files:
        print("验证集中没有找到图像文件")
        return

    print(f"\n可视化 {len(image_files)} 张图像的预测结果...")

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(val_images_dir, image_file)

        try:
            # 进行预测
            results = model.predict(
                source=image_path,
                conf=0.25,
                save=True,  # 保存预测结果
                save_txt=True,  # 保存标签文件
                project="runs/detect",
                name="predict"
            )

            print(f"图像 {i + 1}/{len(image_files)}: {image_file}")

            # 显示预测信息
            for r in results:
                if r.boxes is not None:
                    print(f"  检测到 {len(r.boxes)} 个对象")
                    for j, box in enumerate(r.boxes):
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id] if hasattr(model, 'names') else f'Class_{class_id}'
                        confidence = box.conf[0]
                        print(f"    对象 {j + 1}: {class_name} (置信度: {confidence:.3f})")
                else:
                    print("  未检测到对象")

        except Exception as e:
            print(f"预测图像 {image_file} 时出错: {e}")


def check_dataset_config():
    """检查数据集配置"""
    import yaml

    print("\n=== 检查数据集配置 ===")

    try:
        with open("../cfg/ssdd.yaml", 'r') as f:
            config = yaml.safe_load(f)

        print(f"数据集路径: {config.get('path', '未找到')}")
        print(f"训练集: {config.get('train', '未找到')}")
        print(f"验证集: {config.get('val', '未找到')}")
        print(f"类别数量: {config.get('nc', '未找到')}")
        print(f"类别名称: {config.get('names', '未找到')}")

        # 检查路径是否存在
        base_path = config.get('path', '')
        val_path = config.get('val', '')

        if base_path and val_path:
            full_val_path = os.path.join(base_path, val_path)
            if os.path.exists(full_val_path):
                val_images = [f for f in os.listdir(full_val_path)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                print(f"验证集图像数量: {len(val_images)}")
            else:
                print(f"验证集路径不存在: {full_val_path}")

    except Exception as e:
        print(f"检查数据集配置时出错: {e}")


if __name__ == "__main__":
    print("开始验证训练好的模型...")

    # 1. 检查数据集配置
    check_dataset_config()

    # 2. 验证模型
    validate_trained_model()

    # 3. 可视化预测结果
    visualize_predictions()

    print("\n验证完成!")