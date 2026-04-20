import os
import random
import shutil
from pathlib import Path

def extract_and_split(src_dir, dst_dir, total_num=None, train_ratio=0.8, move=False, seed=42):
    """
    从源文件夹随机提取指定数量的图片，并按比例划分为训练集和验证集

    参数:
        src_dir: 源图片文件夹路径
        dst_dir: 输出根目录（将在其中创建 train/ 和 val/ 子文件夹）
        total_num: 要提取的图片总数（整数），若为 None 则提取全部图片
        train_ratio: 训练集比例，默认 0.8
        move: 是否移动文件（True）还是复制（False），默认复制
        seed: 随机种子，用于可复现的划分
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    # 检查源文件夹
    if not src_path.exists() or not src_path.is_dir():
        print(f"错误：源文件夹不存在或不是目录: {src_dir}")
        return

    # 支持的图片扩展名
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff',
                  '.JPG', '.JPEG', '.PNG', '.BMP', '.TIF', '.TIFF')
    all_images = [f for f in src_path.iterdir() if f.suffix in image_exts]

    if not all_images:
        print("源文件夹中没有找到图片文件。")
        return

    total_available = len(all_images)

    # 确定要提取的数量
    if total_num is None:
        total_num = total_available
        print(f"未指定提取数量，将使用全部 {total_num} 张图片")
    else:
        total_num = int(total_num)
        if total_num <= 0:
            print("提取数量必须大于 0")
            return
        if total_num > total_available:
            print(f"提取数量 {total_num} 超过总图片数 {total_available}，将提取全部图片。")
            total_num = total_available

    # 随机抽取
    random.seed(seed)
    selected_images = random.sample(all_images, total_num)

    # 随机打乱（用于划分）
    random.shuffle(selected_images)

    # 计算训练集数量
    train_count = int(round(total_num * train_ratio))
    if train_count == 0:
        print(f"训练集比例 {train_ratio} 过低，导致训练集为空。")
        return
    if train_count == total_num:
        print(f"训练集比例 {train_ratio} 过高，导致验证集为空。")
        return

    train_images = selected_images[:train_count]
    val_images = selected_images[train_count:]

    # 创建输出文件夹
    train_dir = dst_path / 'train'
    val_dir = dst_path / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    def process_images(images, target_dir, set_name):
        for img in images:
            dst_file = target_dir / img.name
            try:
                if move:
                    shutil.move(str(img), str(dst_file))
                else:
                    shutil.copy2(str(img), str(dst_file))
                print(f"{'移动' if move else '复制'} {set_name}: {img.name}")
            except Exception as e:
                print(f"处理 {img.name} 时出错: {e}")

    process_images(train_images, train_dir, 'train')
    process_images(val_images, val_dir, 'val')

    print(f"\n划分完成！")
    print(f"总提取图片数: {total_num}")
    print(f"训练集: {len(train_images)} 张 ({len(train_images)/total_num:.1%}) -> {train_dir}")
    print(f"验证集: {len(val_images)} 张 ({len(val_images)/total_num:.1%}) -> {val_dir}")

if __name__ == "__main__":
    # ========== 配置参数 ==========
    source_folder = r"D:\Study\10_dataset\HRSID_SPLIT\offshore"   # 源图片文件夹
    output_folder = r"D:\Study\10_dataset\HRSID_SPLIT\offshore_train"          # 输出根目录（将创建 train/ 和 val/）
    extract_num = 535                        # 要提取的图片数量（None 表示提取全部）
    train_ratio = 0.8                         # 训练集比例（8:2）
    move_files = False                        # True: 移动文件，False: 复制文件
    random_seed = 42                          # 随机种子

    # ========== 执行提取与划分 ==========
    extract_and_split(source_folder, output_folder,
                      total_num=extract_num,
                      train_ratio=train_ratio,
                      move=move_files,
                      seed=random_seed)