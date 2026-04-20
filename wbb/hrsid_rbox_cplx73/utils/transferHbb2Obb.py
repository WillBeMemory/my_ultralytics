import cv2
import numpy as np
import os
from pathlib import Path

def hbb_to_obb(image, hbb_pixel, return_deg=True):
    """
    将水平边界框 (HBB) 转换为有向边界框 (OBB)

    参数:
        image: np.ndarray, 原始SAR图像（灰度或彩色）
        hbb_pixel: tuple, (x1, y1, x2, y2) 像素坐标（左上角和右下角）
        return_deg: bool, 是否返回角度为度数（True）或弧度（False）

    返回:
        obb_pixel: tuple, (cx, cy, w, h, angle)
            cx, cy: OBB 中心点像素坐标
            w, h: OBB 的宽度和高度（像素）
            angle: 旋转角度，默认度（范围[-90,0)），弧度若 return_deg=False
    """
    x1, y1, x2, y2 = hbb_pixel
    # 确保坐标整数且在图像范围内
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(image.shape[1], int(x2))
    y2 = min(image.shape[0], int(y2))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("无效的裁剪区域")

    roi = image[y1:y2, x1:x2]

    # 灰度化 + OTSU 二值化
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学处理：开运算去噪 + 闭运算填充孔洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 提取最大轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # 无轮廓时返回原始 HBB（角度为0）
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        angle = 0.0
        return (cx, cy, w, h, angle) if return_deg else (cx, cy, w, h, 0.0)

    max_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(max_contour)          # ((cx_roi, cy_roi), (w, h), angle)
    (cx_roi, cy_roi), (w_roi, h_roi), angle = rect

    # 角度转换：cv2.minAreaRect 返回的是 [-90,0) 度，这里保持原样
    if not return_deg:
        angle = np.deg2rad(angle)

    # 映射回原图坐标
    cx_orig = cx_roi + x1
    cy_orig = cy_roi + y1

    return (cx_orig, cy_orig, w_roi, h_roi, angle)


def convert_yolo_to_obb(image_path, label_path, output_path, class_mapping=None):
    """
    将单个图片的 YOLO 标注文件转换为 OBB 标注文件（YOLOv8 OBB 格式）

    参数:
        image_path: str, 图片路径
        label_path: str, 原始 YOLO 标注文件路径（.txt）
        output_path: str, 输出 OBB 标注文件路径
        class_mapping: dict, 类别映射（可选），若 None 则保持原 class_id
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"警告：无法读取图片 {image_path}，跳过")
        return

    h_img, w_img = img.shape[:2]

    # 读取 YOLO 标注
    with open(label_path, 'r') as f:
        lines = f.readlines()

    obb_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"警告：标注格式错误（需要5个值）: {line.strip()}")
            continue

        class_id = int(parts[0])
        if class_mapping is not None:
            class_id = class_mapping.get(class_id, class_id)

        # YOLO 归一化坐标：x_center, y_center, width, height
        cx_norm = float(parts[1])
        cy_norm = float(parts[2])
        w_norm = float(parts[3])
        h_norm = float(parts[4])

        # 转换为像素坐标（HBB 的左上角、右下角）
        x1 = (cx_norm - w_norm / 2) * w_img
        y1 = (cy_norm - h_norm / 2) * h_img
        x2 = (cx_norm + w_norm / 2) * w_img
        y2 = (cy_norm + h_norm / 2) * h_img

        try:
            # 调用转换函数，返回 OBB 像素坐标和角度（度）
            cx_obb, cy_obb, w_obb, h_obb, angle_deg = hbb_to_obb(img, (x1, y1, x2, y2), return_deg=True)
        except Exception as e:
            print(f"转换失败 (图片: {image_path}, 框: {line.strip()}) 错误: {e}")
            # 失败时使用原始 HBB，角度为 0
            cx_obb = (x1 + x2) / 2.0
            cy_obb = (y1 + y2) / 2.0
            w_obb = x2 - x1
            h_obb = y2 - y1
            angle_deg = 0.0

        # 归一化 OBB 坐标（YOLO OBB 格式要求归一化中心点和宽高，角度用度）
        cx_norm_obb = cx_obb / w_img
        cy_norm_obb = cy_obb / h_img
        w_norm_obb = w_obb / w_img
        h_norm_obb = h_obb / h_img

        # 格式：class_id cx_norm cy_norm w_norm h_norm angle_deg
        obb_lines.append(f"{class_id} {cx_norm_obb:.6f} {cy_norm_obb:.6f} {w_norm_obb:.6f} {h_norm_obb:.6f} {angle_deg:.6f}\n")

    # 写入输出文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.writelines(obb_lines)
    print(f"已转换: {label_path} -> {output_path} (共 {len(obb_lines)} 个目标)")


def batch_convert(image_folder, label_folder, output_folder, class_mapping=None, image_exts=('.jpg','.jpeg','.png','.tif','.bmp')):
    """
    批量转换文件夹下的所有图片和标注

    参数:
        image_folder: str, 图片文件夹路径
        label_folder: str, YOLO 标注文件夹路径（标注文件与图片同名，扩展名为 .txt）
        output_folder: str, 输出 OBB 标注文件夹路径
        class_mapping: dict, 类别映射（可选）
        image_exts: tuple, 支持的图片扩展名
    """
    # 收集所有图片文件
    image_files = []
    for ext in image_exts:
        image_files.extend(Path(image_folder).glob(f"*{ext}"))
        image_files.extend(Path(image_folder).glob(f"*{ext.upper()}"))

    if not image_files:
        print("未找到任何图片文件。")
        return

    for img_path in image_files:
        stem = img_path.stem
        label_path = Path(label_folder) / f"{stem}.txt"
        if not label_path.exists():
            print(f"警告：未找到标注文件 {label_path}，跳过图片 {img_path}")
            continue

        output_path = Path(output_folder) / f"{stem}.txt"
        convert_yolo_to_obb(str(img_path), str(label_path), str(output_path), class_mapping)


if __name__ == "__main__":
    # 配置路径（请根据实际修改）
    image_dir = r"D:\Study\10_dataset\HRSID_YOLO\images\val"      # 图片文件夹
    label_dir = r"D:\Study\10_dataset\HRSID_YOLO\labels\val"      # YOLO 标注文件夹（.txt）
    output_dir = r"D:\Study\10_dataset\HRSID_YOLO\obb\labels\val" # 输出 OBB 标注文件夹

    # 可选：类别映射（例如将原始类别 0 映射为 0，保持不变；可自定义）
    class_map = None   # 或 {0:0, 1:1, ...}

    # 执行批量转换
    batch_convert(image_dir, label_dir, output_dir, class_mapping=class_map)