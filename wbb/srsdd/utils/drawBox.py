import cv2
import numpy as np
from pathlib import Path

# ==================== 用户可修改参数 ====================
IMAGE_FOLDER = r"D:\Study\10_dataset\SRSDD_YOLO\images\train"   # 图像文件夹路径
LABEL_FOLDER = r"D:\Study\10_dataset\SRSDD_YOLO\labels\train"   # 标注文件夹路径（与图像同名 .txt）
OUTPUT_FOLDER = r"D:\Study\10_dataset\SRSDD_YOLO\vis_train"     # 输出文件夹路径

# 类别名称列表，索引对应 class_id，例如 ['ship', 'container', 'dredger']
# 若为 None，则只显示 class_id 数字
CLASS_NAMES = ['ship']   # 根据你的类别列表修改

# 边框颜色 (B, G, R)，例如绿色 (0,255,0)，红色 (0,0,255)
COLOR = (0, 255, 0)      # 绿色

# 线条粗细
THICKNESS = 2
# =======================================================

def draw_yolo_obb(image_path, label_path, output_path, class_names=None, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制 YOLO OBB 格式的旋转框。

    Args:
        image_path (str): 输入图像路径
        label_path (str): 对应的 YOLO OBB 标注文件路径
        output_path (str): 输出图像保存路径
        class_names (list): 类别名称列表，索引对应 class_id；若为 None 则只显示 id
        color (tuple): 边框颜色 (B,G,R)
        thickness (int): 边框线条粗细
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
    h, w = img.shape[:2]

    # 读取标注文件
    if not Path(label_path).exists():
        print(f"标注文件不存在: {label_path}")
        cv2.imwrite(output_path, img)
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 9:  # class_id + 8个坐标
            print(f"格式错误: {line}")
            continue

        class_id = int(parts[0])
        # 归一化坐标转换为像素坐标
        coords = list(map(float, parts[1:9]))
        points = []
        for i in range(0, 8, 2):
            x = int(coords[i] * w)
            y = int(coords[i+1] * h)
            points.append((x, y))
        points = np.array(points, dtype=np.int32)

        # 绘制多边形
        cv2.polylines(img, [points], isClosed=True, color=color, thickness=thickness)

        # 添加标签文本
        label_text = str(class_id) if class_names is None else class_names[class_id]
        # 将文本放在第一个点附近
        pos = points[0]
        cv2.putText(img, label_text, (pos[0], pos[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness-1)

    # 保存结果
    cv2.imwrite(output_path, img)
    print(f"已保存: {output_path}")

def process_folder(image_folder, label_folder, output_folder, class_names=None, color=(0,255,0), thickness=2):
    """
    批量处理文件夹中的图像和标注。

    Args:
        image_folder (str): 图像文件夹路径
        label_folder (str): 标注文件夹路径（应与图像同名，扩展名为 .txt）
        output_folder (str): 输出文件夹路径
        class_names (list): 类别名称列表
        color (tuple): 边框颜色
        thickness (int): 线条粗细
    """
    image_folder = Path(image_folder)
    label_folder = Path(label_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # 获取所有图像文件（支持常见格式）
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_paths = [p for p in image_folder.iterdir() if p.suffix.lower() in image_extensions]

    if not image_paths:
        print(f"未找到图像文件: {image_folder}")
        return

    for img_path in image_paths:
        # 构建对应的标注路径
        label_path = label_folder / (img_path.stem + '.txt')
        output_path = output_folder / img_path.name
        draw_yolo_obb(str(img_path), str(label_path), str(output_path), class_names, color, thickness)

if __name__ == "__main__":
    process_folder(IMAGE_FOLDER, LABEL_FOLDER, OUTPUT_FOLDER, CLASS_NAMES, COLOR, THICKNESS)