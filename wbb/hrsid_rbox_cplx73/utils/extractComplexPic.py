import os
import shutil
import cv2
import numpy as np


def is_nearshore(image_path, brightness_threshold=52, highlight_ratio=0.15):
    """
    判断一张SAR图像是否为近岸图像（复杂背景）。
    判断依据：图像中灰度值 ≥ brightness_threshold 的像素占比是否超过 highlight_ratio。

    参数：
        image_path: 图像文件路径
        brightness_threshold: 亮度阈值，高于此值视为“白色/高亮”元素
        highlight_ratio: 高亮像素占比阈值，超过此值判定为近岸

    返回：
        True 表示近岸图像，False 表示海面图像
    """
    # 以灰度模式读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"警告：无法读取图像 {image_path}，跳过")
        return False

    # 计算高亮像素数量及占比
    highlight_pixels = np.sum(img > brightness_threshold)
    total_pixels = img.size
    ratio = highlight_pixels / total_pixels

    return ratio > highlight_ratio


def copy_nearshore_images(src_folder, dst_folder, brightness_threshold=52, highlight_ratio=0.15):
    """
    将源文件夹中的近岸图像复制到目标文件夹。

    参数：
        src_folder: 原始图像文件夹路径
        dst_folder: 目标文件夹路径（用于存放近岸图像）
        brightness_threshold: 亮度阈值
        highlight_ratio: 高亮像素占比阈值
    """
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # 支持的图像格式（可根据实际情况扩展）
    image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(src_folder):
        if filename.lower().endswith(image_extensions):
            file_path = os.path.join(src_folder, filename)
            if is_nearshore(file_path, brightness_threshold, highlight_ratio):
                shutil.copy2(file_path, dst_folder)
                print(f"✅ 复制近岸图像: {filename}")
            else:
                print(f"⏭️ 跳过海面图像: {filename}")


def analyze_highlight_ratios(folder, brightness_threshold=52):
    """
    分析文件夹中所有图像的高亮像素占比，帮助确定合适的 highlight_ratio 阈值。
    打印每张图的文件名和占比，并返回所有占比的列表。
    """
    ratios = []
    image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')

    for filename in os.listdir(folder):
        if filename.lower().endswith(image_extensions):
            file_path = os.path.join(folder, filename)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            ratio = np.sum(img > brightness_threshold) / img.size
            ratios.append((filename, ratio))
            print(f"{filename}: {ratio:.4f}")

    # 按占比升序排序
    ratios.sort(key=lambda x: x[1])
    print("\n--- 排序后 ---")
    for f, r in ratios:
        print(f"{f}: {r:.4f}")

    return ratios


if __name__ == "__main__":
    # ==================== 配置区域 ====================
    # 请根据实际情况修改以下路径和参数
    source_directory = r"D:\Study\10_dataset\HRSID_YOLO\images\train"  # 原始图像文件夹
    target_directory = r"D:\Study\10_dataset\HRSID_YOLO_COMPLEX\images\train"  # 目标文件夹（用于存放近岸图像）

    # 亮度阈值：根据你观察到的 #343434 对应的灰度值 52 设定
    brightness_thresh = 34  # 取值范围 0-255（对于8位图像）

    # 高亮像素占比阈值：需要根据分析结果确定，建议先用 analyze_highlight_ratios 查看分布
    highlight_thresh = 0.15  # 初始值设为0.15，后续可调整

    # 是否先运行分析模式（True 则只分析占比，不复制；False 则直接复制）
    run_analysis = False  # 初次运行时建议设为 True，观察数据分布
    # ==================================================

    if run_analysis:
        print("=== 分析高亮像素占比 ===")
        analyze_highlight_ratios(source_directory, brightness_thresh)
    else:
        print("=== 开始复制近岸图像 ===")
        copy_nearshore_images(source_directory, target_directory,
                              brightness_thresh, highlight_thresh)
        print("处理完成！")