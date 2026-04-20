import os
import cv2
import shutil
from pathlib import Path

def is_deep_sea(image_path, threshold=50):
    """
    判断图像是否为远海（背景较暗）
    参数:
        image_path: 图片路径
        threshold: 灰度阈值，低于此值视为远海（默认 50，对应约 #323232）
    返回:
        True: 远海，False: 近岸
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"警告：无法读取图片 {image_path}，跳过")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_gray = cv2.mean(gray)[0]
    return mean_gray < threshold

def filter_images(image_folder, threshold=50, remove=False, move_to=None):
    """
    筛选图片，删除或复制远海图片
    参数:
        image_folder: 图片所在文件夹
        threshold: 灰度阈值
        remove: 是否直接删除（True）或复制到其他文件夹（False，此时需要指定 move_to）
        move_to: 若 remove=False，则将远海图片复制到此文件夹（需提供路径）
    """
    image_folder = Path(image_folder)
    image_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.JPG', '.JPEG', '.PNG', '.TIF', '.TIFF', '.BMP')
    images = [f for f in image_folder.iterdir() if f.suffix in image_exts]

    if not images:
        print("未找到任何图片文件。")
        return

    deep_sea_images = []
    for img_path in images:
        if is_deep_sea(img_path, threshold):
            deep_sea_images.append(img_path)
            print(f"远海: {img_path.name}")
        else:
            print(f"近岸: {img_path.name}")

    if not deep_sea_images:
        print("未发现远海图片。")
        return

    if remove:
        # 删除模式
        for img_path in deep_sea_images:
            try:
                img_path.unlink()
                print(f"已删除: {img_path.name}")
            except Exception as e:
                print(f"删除失败 {img_path.name}: {e}")
    else:
        # 复制模式
        if move_to is None:
            print("错误：当 remove=False 时，必须提供 move_to 参数。")
            return
        move_to = Path(move_to)
        move_to.mkdir(parents=True, exist_ok=True)
        for img_path in deep_sea_images:
            try:
                # 使用 copy2 保留文件元数据
                shutil.copy2(str(img_path), str(move_to / img_path.name))
                print(f"已复制: {img_path.name} -> {move_to}")
            except Exception as e:
                print(f"复制失败 {img_path.name}: {e}")

if __name__ == "__main__":
    # ========== 配置参数 ==========
    image_dir = r"D:\Study\10_dataset\all"          # 图片文件夹路径
    threshold = 23                                  # 灰度阈值（低于此值视为远海）
    remove_mode = False                             # True: 直接删除远海图片；False: 复制到指定文件夹
    move_to_dir = r"D:\Study\10_dataset\del"         # 当 remove_mode=False 时，远海图片复制到此文件夹

    # ========== 执行筛选 ==========
    filter_images(
        image_folder=image_dir,
        threshold=threshold,
        remove=remove_mode,
        move_to=move_to_dir if not remove_mode else None
    )