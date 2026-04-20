import random
import shutil
from pathlib import Path
from typing import List, Tuple

def build_dataset(
    label_dir: str,
    image_dir: str,
    output_dir: str = None,
    ratio: float = 0.65,
    copy_files: bool = True,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    根据标注文件夹和图片文件夹，按比例构建数据集（图片不重复）。

    Args:
        label_dir: 文件夹1，存放标注文件（如 .txt, .xml）。文件名（不含扩展名）视为已存在图片名。
        image_dir: 文件夹2，存放图片文件（如 .jpg, .png）。从中选取不重复的图片补充。
        output_dir: 输出目录（可选）。若提供，则将选中的图片复制到此目录下。
        ratio: 文件夹1的图片数量占最终数据集的比例（默认 0.7）。
        copy_files: 是否将选中的图片复制到 output_dir（若 output_dir 不为 None）。
        seed: 随机种子，保证可复现。

    Returns:
        (list1, list2): list1 为文件夹1中所有图片的路径（原路径），
                        list2 为从文件夹2中选中的图片路径（原路径）。
    """
    # 设置随机种子
    random.seed(seed)

    # 将路径转换为 Path 对象
    label_dir = Path(label_dir)
    image_dir = Path(image_dir)

    if not label_dir.is_dir():
        raise NotADirectoryError(f"标注文件夹不存在: {label_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"图片文件夹不存在: {image_dir}")

    # 1. 从文件夹1中获取图片名集合（标注文件对应的图片名）
    existing_names = set()
    for ext in ['*.txt', '*.xml']:   # 可根据实际标注扩展名调整
        for file in label_dir.glob(ext):
            existing_names.add(file.stem)   # 不含扩展名

    # 2. 从文件夹2中读取所有图片文件（支持常见图片格式）
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    all_images = [p for p in image_dir.iterdir() if p.suffix.lower() in img_extensions]

    # 3. 过滤掉与文件夹1中同名的图片
    available_images = [p for p in all_images if p.stem not in existing_names]

    # 4. 计算需要从文件夹2中选取的数量
    num_from_label = len(existing_names)
    # 目标比例： num_from_label / (num_from_label + num_from_image) = ratio
    # => num_from_image = num_from_label * (1 - ratio) / ratio
    num_from_image = int(round(num_from_label * (1 - ratio) / ratio))

    if num_from_image > len(available_images):
        print(f"警告：文件夹2中可用图片不足（需要 {num_from_image}，实际 {len(available_images)}），将使用全部可用图片。")
        num_from_image = len(available_images)

    # 随机选择
    selected_from_image = random.sample(available_images, num_from_image) if num_from_image > 0 else []

    # 5. 输出结果
    from_label_list = [p for p in label_dir.iterdir() if p.suffix.lower() in img_extensions]  # 假设文件夹1中也有图片？通常只有标注文件，但如果有图片也可以直接使用
    # 更合理：文件夹1中可能没有图片，只有标注。这里我们只返回图片路径（实际使用时需要自行映射）。
    # 为了方便，我们返回图片路径，如果文件夹1中也有图片，则包括它们；否则只返回选中图片。
    # 我们假设文件夹1中不包含图片，所以从文件夹1中返回空列表或仅返回标注名？根据问题，我们只关心图片。
    # 根据需求“最终文件夹1的数量占70”，文件夹1的图片是指与标注文件对应的图片，不一定在文件夹1内。
    # 因此，实际使用时，用户应自行准备与标注同名的图片。这里我们只返回选中的图片路径（来自文件夹2）。
    # 但为了符合返回值，我们返回两个列表：文件夹1对应的图片路径（假设在某个位置）和文件夹2选中的路径。
    # 这里简化：假设文件夹1的图片已经存在于某个位置，由用户自己管理。我们只返回文件夹1的图片名列表，供用户参考。

    # 更实用的做法：直接返回从文件夹1中需要保留的图片列表（通过标注名确定）和从文件夹2中选中的图片列表。
    # 用户可根据这些列表复制或移动文件。
    # 我们返回文件名列表（不含路径），并可选复制到输出目录。

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 复制文件夹1的图片（需要提供图片源路径，这里假设图片在某个地方，用户需自行指定）
        # 由于用户未提供文件夹1的图片路径，我们无法复制。因此我们只复制从文件夹2选中的图片。
        for src in selected_from_image:
            dst = output_dir / src.name
            if copy_files:
                shutil.copy2(src, dst)
            else:
                print(f"将链接/移动: {src} -> {dst}")
        print(f"已将 {len(selected_from_image)} 张图片复制到 {output_dir}")

    # 返回文件夹1的图片名列表和文件夹2选中的图片路径列表
    label_names = list(existing_names)
    return label_names, selected_from_image

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 设置路径（请修改为实际路径）
    label_dir = r"D:\Study\10_dataset\HRSID_COMPLEX\labels\all"   # 存放标注文件（.xml或.txt）
    image_dir = r"D:\Study\10_dataset\HRSID_YOLO\images\all"          # 存放图片文件
    output_dir = r"D:\Study\10_dataset\HRSID_COMPLEX\labels\others"            # 输出目录（可选）

    label_names, selected_images = build_dataset(
        label_dir=label_dir,
        image_dir=image_dir,
        output_dir=output_dir,
        ratio=0.7,
        copy_files=True,
        seed=42
    )

    print(f"文件夹1中的图片数量（根据标注文件）: {len(label_names)}")
    print(f"从文件夹2中选取的图片数量: {len(selected_images)}")
    print("文件夹1中的前5个图片名:", label_names[:5])
    print("从文件夹2中选取的前5个图片:", [str(p) for p in selected_images[:5]])