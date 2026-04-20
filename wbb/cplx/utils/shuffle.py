import os
import shutil
import random
import argparse

def split_dataset_all_to_train_test(
    images_source_dir,
    labels_source_dir,
    output_base_dir,
    train_ratio=0.8,
    seed=42,
    image_extensions=None
):
    """
    将 images/all 和 labels/all 中的数据集按比例随机划分为训练集和测试集，
    输出到 output_base_dir/images/train, output_base_dir/images/test,
    output_base_dir/labels/train, output_base_dir/labels/test。

    参数：
        images_source_dir: 源图片文件夹路径（例如 datasets/images/all）
        labels_source_dir: 源标签文件夹路径（例如 datasets/labels/all）
        output_base_dir: 输出根目录（例如 datasets），将在其下创建 images/train, images/test, labels/train, labels/test
        train_ratio: 训练集比例，默认0.8
        seed: 随机种子，确保可重复性
        image_extensions: 支持的图片扩展名元组，默认常见格式
    """
    # 默认图片扩展名
    if image_extensions is None:
        image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')

    # 设置随机种子
    random.seed(seed)

    # 检查源目录
    if not os.path.isdir(images_source_dir):
        raise ValueError(f"图片源目录不存在: {images_source_dir}")
    if not os.path.isdir(labels_source_dir):
        raise ValueError(f"标签源目录不存在: {labels_source_dir}")

    # 收集所有有效的图片-标签对
    items = []          # 每个元素为 (图片路径, 标签路径)
    missing_labels = [] # 记录缺失标签的图片

    for fname in os.listdir(images_source_dir):
        if fname.lower().endswith(image_extensions):
            base = os.path.splitext(fname)[0]
            img_path = os.path.join(images_source_dir, fname)
            lbl_path = os.path.join(labels_source_dir, base + '.txt')

            if os.path.isfile(lbl_path):
                items.append((img_path, lbl_path))
            else:
                missing_labels.append(fname)

    # 输出缺失警告
    if missing_labels:
        print("警告：以下图片缺少对应的标签文件，将被忽略：")
        for f in missing_labels:
            print(f"  {f}")

    if not items:
        print("错误：没有找到任何有效的图片-标签对。")
        return

    # 随机打乱
    random.shuffle(items)

    # 计算分割索引
    split_idx = int(len(items) * train_ratio)
    train_items = items[:split_idx]
    test_items = items[split_idx:]

    print(f"总有效样本数: {len(items)}")
    print(f"训练集样本数: {len(train_items)} ({train_ratio*100:.1f}%)")
    print(f"测试集样本数: {len(test_items)} ({(1-train_ratio)*100:.1f}%)")

    # 定义目标子目录
    subsets = {
        'train': train_items,
        'test': test_items
    }

    # 创建目标目录并复制文件
    for subset_name, subset_items in subsets.items():
        images_target = os.path.join(output_base_dir, 'images', subset_name)
        labels_target = os.path.join(output_base_dir, 'labels', subset_name)

        os.makedirs(images_target, exist_ok=True)
        os.makedirs(labels_target, exist_ok=True)

        for img_path, lbl_path in subset_items:
            # 复制图片
            shutil.copy2(img_path, images_target)
            # 复制标签
            shutil.copy2(lbl_path, labels_target)

        print(f"{subset_name}集: 已复制 {len(subset_items)} 对文件到 {images_target} 和 {labels_target}")

    print("数据集划分完成！")


if __name__ == "__main__":
    # ==================== 配置区域（直接修改变量即可）====================
    # 源目录
    source_images = r"D:\Study\10_dataset\HRSID_YOLO_COMPLEX\images\all"   # 你的原始图片文件夹
    source_labels = r"D:\Study\10_dataset\HRSID_YOLO_COMPLEX\labels\all"   # 你的原始标签文件夹
    # 输出根目录（会在其下创建 images/train, images/test, labels/train, labels/test）
    output_base = r"D:\Study\10_dataset\HRSID_COMPLEX"                 # 例如 datasets

    train_ratio = 0.8        # 训练集比例
    random_seed = 42         # 随机种子
    # ===================================================================

    # 运行划分
    split_dataset_all_to_train_test(
        images_source_dir=source_images,
        labels_source_dir=source_labels,
        output_base_dir=output_base,
        train_ratio=train_ratio,
        seed=random_seed
    )