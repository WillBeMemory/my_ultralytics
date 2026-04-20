import shutil
from pathlib import Path

# ==================== 用户配置区域 ====================
# 数据集根目录（根据你的实际路径修改）
DATASET_ROOT = Path(r"D:\Study\10_dataset\HRSID_YOLO_RBOX")  # 你的数据集根目录

# 图像子目录（相对 DATASET_ROOT）
IMAGE_TRAIN_DIR = "images/train"
IMAGE_VAL_DIR   = "images/val"

# 原始标注文件所在目录（相对 DATASET_ROOT）
# 如果所有标注文件都在一个文件夹里（如 labels_original），请填写该路径
LABEL_SOURCE_DIR = "labels/all"   # 如果标注文件已分散在 labels/train 和 labels/val，可不使用

# 目标标注输出目录（相对 DATASET_ROOT）
LABEL_TARGET_TRAIN = "labels/train"
LABEL_TARGET_VAL   = "labels/val"

# 操作模式：'move' 或 'copy' （推荐先使用 'copy' 保留原始文件）
ACTION = 'copy'   # 或 'move'

# 如果找不到对应的标注文件，是否报错（True: 报错，False: 跳过并打印警告）
RAISE_IF_MISSING = False
# ===================================================

def split_labels_by_images(image_dir, label_source_dir, target_label_dir, action):
    """
    根据 image_dir 中的图片，将对应的标注文件从 label_source_dir 复制/移动到 target_label_dir。
    """
    # 创建目标目录
    target_label_dir.mkdir(parents=True, exist_ok=True)

    # 支持的图片扩展名
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    # 遍历所有图片文件
    for img_path in image_dir.glob("*"):
        if img_path.suffix.lower() not in img_extensions:
            continue   # 跳过非图片文件

        # 提取图片名（不含扩展名）
        stem = img_path.stem

        # 对应的标注文件路径
        src_label = label_source_dir / f"{stem}.txt"

        if not src_label.exists():
            msg = f"警告：标注文件不存在 {src_label}"
            if RAISE_IF_MISSING:
                raise FileNotFoundError(msg)
            else:
                print(msg)
                continue

        # 目标路径
        dst_label = target_label_dir / src_label.name

        # 执行复制或移动
        if action == 'copy':
            shutil.copy2(src_label, dst_label)
        elif action == 'move':
            shutil.move(str(src_label), str(dst_label))
        else:
            raise ValueError("action 必须是 'copy' 或 'move'")

        print(f"{action}: {src_label.name} -> {dst_label}")

def main():
    # 构建完整路径
    img_train_dir = DATASET_ROOT / IMAGE_TRAIN_DIR
    img_val_dir   = DATASET_ROOT / IMAGE_VAL_DIR
    label_source  = DATASET_ROOT / LABEL_SOURCE_DIR
    label_train   = DATASET_ROOT / LABEL_TARGET_TRAIN
    label_val     = DATASET_ROOT / LABEL_TARGET_VAL

    # 检查图片目录是否存在
    if not img_train_dir.exists():
        print(f"错误：训练图片目录不存在 {img_train_dir}")
        return
    if not img_val_dir.exists():
        print(f"错误：验证图片目录不存在 {img_val_dir}")
        return

    # 如果原始标注目录不存在，尝试默认 labels/ 作为源目录
    if not label_source.exists():
        print(f"警告：标注源目录 {label_source} 不存在，将尝试使用默认 labels/ 目录")
        # 若源目录不存在，可尝试从根目录下的 labels/ 查找
        label_source = DATASET_ROOT / "labels"
        if not label_source.exists():
            print(f"错误：无法找到标注文件目录 {label_source}")
            return

    print("正在处理训练集图片...")
    split_labels_by_images(img_train_dir, label_source, label_train, ACTION)

    print("\n正在处理验证集图片...")
    split_labels_by_images(img_val_dir, label_source, label_val, ACTION)

    print("\n完成！")

if __name__ == "__main__":
    main()