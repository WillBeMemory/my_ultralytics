import cv2
import numpy as np
import math
import xml.etree.ElementTree as ET
from pathlib import Path

def draw_rotated_rect(img, cx, cy, w, h, angle_rad, color=(0,255,0), thickness=2):
    """
    在图像上绘制旋转矩形（基于中心点、宽、高、角度）
    """
    # 矩形四个角点相对于中心点的坐标（未旋转）
    half_w = w / 2.0
    half_h = h / 2.0
    points = np.array([
        [-half_w, -half_h],
        [ half_w, -half_h],
        [ half_w,  half_h],
        [-half_w,  half_h]
    ], dtype=np.float32)

    # 旋转矩阵
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    # 旋转点集并平移到中心
    rotated = np.dot(points, rot_matrix.T) + np.array([cx, cy])
    rotated = rotated.astype(np.int32)

    # 绘制多边形
    cv2.polylines(img, [rotated], isClosed=True, color=color, thickness=thickness)

def visualize_xml(xml_path, img_dir, output_path, color=(0,255,0), thickness=2):
    """
    读取XML并在图像上绘制所有旋转框，保存结果
    """
    # 解析XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取图像文件名（假设图像扩展名为 .jpg 或 .png）
    filename_elem = root.find('filename')
    if filename_elem is None:
        print(f"XML缺少<filename>标签: {xml_path}")
        return
    img_name = filename_elem.text
    # 尝试多种扩展名
    img_path = None
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        candidate = img_dir / (Path(img_name).stem + ext)
        if candidate.exists():
            img_path = candidate
            break
    if img_path is None:
        print(f"未找到图像: {img_name} 在 {img_dir}")
        return

    # 读取图像
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"无法读取图像: {img_path}")
        return

    # 获取图像尺寸（XML中也有，但直接使用实际尺寸）
    h, w = img.shape[:2]

    # 遍历所有object
    for obj in root.findall('object'):
        name = obj.find('name').text
        robndbox = obj.find('robndbox')
        if robndbox is None:
            continue

        cx = float(robndbox.find('cx').text)
        cy = float(robndbox.find('cy').text)
        w_rect = float(robndbox.find('w').text)
        h_rect = float(robndbox.find('h').text)
        angle = float(robndbox.find('angle').text)

        # 绘制旋转矩形（角度已是弧度）
        draw_rotated_rect(img, cx, cy, w_rect, h_rect, angle, color, thickness)

        # 可选：添加文本标签
        cv2.putText(img, name, (int(cx-10), int(cy-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 保存结果
    cv2.imwrite(str(output_path), img)
    print(f"已保存: {output_path}")

def main():
    # ========== 用户配置 ==========
    XML_DIR = Path(r"D:\Study\10_dataset\HRSID_RBox_Annotations\HRSID_RBox_Annotations")   # XML文件目录
    IMAGE_DIR = Path(r"D:\Study\10_dataset\HRSID_YOLO\images\all")       # 图像文件目录
    OUTPUT_DIR = Path(r"D:\Study\10_dataset\HRSID_RBox_Annotations")  # 输出目录
    NUM_TO_DRAW = 5                # 只绘制前5张图片（可根据需要改为随机选取）
    # ============================

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 获取所有XML文件
    xml_files = sorted(XML_DIR.glob("*.xml"))  # 排序后取前NUM_TO_DRAW个
    if not xml_files:
        print("未找到XML文件")
        return

    # 只处理前NUM_TO_DRAW个
    for i, xml_file in enumerate(xml_files[:NUM_TO_DRAW]):
        output_path = OUTPUT_DIR / f"{xml_file.stem}_vis.jpg"
        visualize_xml(xml_file, IMAGE_DIR, output_path, color=(0,255,0), thickness=2)

    print(f"完成，已处理 {min(NUM_TO_DRAW, len(xml_files))} 张图片")

if __name__ == "__main__":
    main()