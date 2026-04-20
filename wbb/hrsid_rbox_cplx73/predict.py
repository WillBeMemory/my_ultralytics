import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ==================== 用户可配置参数 ====================
model_path = "./runs/obb/train/weights/best.pt"   # 训练好的 OBB 模型权重
input_dir = r"D:\Study\10_dataset\HRSID_RBOX_CPLX73\images\train"                       # 输入图片文件夹
label_dir = r"D:\Study\10_dataset\HRSID_RBOX_CPLX73\labels\train"                       # 真实标注文件夹（9列坐标格式，与图片同名.txt）
output_dir = r"D:\Study\10_dataset\HRSID_RBOX_CPLX73\predict\train"                     # 输出结果文件夹

# 支持的图片扩展名
img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

# 可视化颜色 (BGR 格式)
color_gt = (0, 255, 0)    # 绿色：真实标注
color_pred = (255, 0, 0)  # 蓝色：预测框

# 置信度阈值（只显示置信度高于此值的预测框）
conf_threshold = 0.25

# 类别名称列表（可选），例如 ['ship']，若为 None 则只显示类别ID
class_names = None   # 或 ['ship', 'container', ...] 根据你的数据集设置
# =======================================================

def draw_obb(img, points, color, thickness=2, label=None):
    """
    在图像上绘制旋转框（多边形）
    points: 四个角点坐标，形状 (4,2) 的 numpy 数组
    """
    cv2.polylines(img, [points], isClosed=True, color=color, thickness=thickness)
    if label is not None:
        # 将文本放在第一个角点附近
        pos = points[0]
        cv2.putText(img, label, (pos[0], pos[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness-1)

def load_ground_truth(label_path, img_shape):
    """
    从 YOLO OBB 格式的 txt 文件加载真实标注框（9列格式：class_id + 8个角点坐标，归一化）。
    返回：列表，每个元素为 (class_id, corners, label_text)
    """
    if not Path(label_path).exists():
        return []
    h, w = img_shape[:2]
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                # 如果不是9列，可能是其他格式，跳过或警告
                continue
            class_id = int(parts[0])
            # 解析8个坐标 (x1,y1,x2,y2,x3,y3,x4,y4)
            coords = list(map(float, parts[1:9]))
            points = []
            for i in range(0, 8, 2):
                x = int(coords[i] * w)
                y = int(coords[i+1] * h)
                points.append((x, y))
            corners = np.array(points, dtype=np.int32)
            # 构造显示文本
            label_text = str(class_id) if class_names is None else class_names[class_id]
            boxes.append((class_id, corners, label_text))
    return boxes

def load_predictions(results, img_shape, conf_thresh):
    """
    从 Ultralytics 预测结果中提取预测框。
    返回：列表，每个元素为 (class_id, corners, confidence, label_text)
    """
    if not results or len(results) == 0:
        return []
    boxes = []
    for r in results:
        if hasattr(r, 'obb') and r.obb is not None:
            obb = r.obb
            if obb.data is not None:
                data = obb.data.cpu().numpy()
                for d in data:
                    conf = d[5]
                    if conf < conf_thresh:
                        continue
                    cls_id = int(d[6])
                    cx, cy, w, h, angle = d[0], d[1], d[2], d[3], d[4]
                    # 将中心点+宽高+角度转换为四个角点
                    corners = yolo_obb_to_corners(cx, cy, w, h, angle)
                    label_text = str(cls_id) if class_names is None else class_names[cls_id]
                    boxes.append((cls_id, corners, conf, label_text))
        elif hasattr(r, 'boxes') and r.boxes is not None:
            for box in r.boxes:
                conf = box.conf.item()
                if conf < conf_thresh:
                    continue
                cls_id = int(box.cls.item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                label_text = str(cls_id) if class_names is None else class_names[cls_id]
                boxes.append((cls_id, corners, conf, label_text))
    return boxes

def yolo_obb_to_corners(x, y, w, h, angle):
    """
    将 YOLO OBB 格式的中心点、宽、高、角度转换为四个角点坐标（像素坐标系）
    angle: 弧度，水平轴到第一边的夹角（顺时针或逆时针取决于训练设置）
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    dx = w / 2
    dy = h / 2
    corners = np.array([[-dx, -dy], [ dx, -dy], [ dx,  dy], [-dx,  dy]])
    rot = np.array([[cos_a, -sin_a], [sin_a,  cos_a]])
    corners = corners @ rot.T
    corners[:, 0] += x
    corners[:, 1] += y
    return corners.astype(np.int32)

def process_image(image_path, model, label_dir, output_dir, conf_threshold=0.25):
    """处理单张图片：加载真实标注、预测、绘制并保存"""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"无法读取图片: {image_path}")
        return
    img_vis = img.copy()
    img_shape = img.shape

    # 1. 绘制真实标注（若标签文件存在）
    if label_dir is not None:
        label_path = Path(label_dir) / (image_path.stem + '.txt')
        gt_boxes = load_ground_truth(label_path, img_shape)
        for _, corners, label_text in gt_boxes:
            draw_obb(img_vis, corners, color_gt, thickness=2, label=label_text)

    # 2. 模型预测
    results = model.predict(str(image_path), imgsz=640, conf=conf_threshold, device=device, verbose=False)

    # 3. 绘制预测框
    pred_boxes = load_predictions(results, img_shape, conf_threshold)
    for _, corners, conf, label_text in pred_boxes:
        display_label = f"{label_text}:{conf:.2f}"
        draw_obb(img_vis, corners, color_pred, thickness=2, label=display_label)

    # 4. 保存结果
    out_path = output_dir / image_path.name
    cv2.imwrite(str(out_path), img_vis)
    print(f"已保存: {out_path}")

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    model.to(device)

    # 遍历输入文件夹下的所有图片
    input_path = Path(input_dir)
    image_files = [p for p in input_path.iterdir() if p.suffix.lower() in img_exts]

    if not image_files:
        print(f"在 {input_dir} 中未找到支持的图片文件 ({img_exts})")
    else:
        print(f"共找到 {len(image_files)} 张图片，开始处理...")
        for img_file in image_files:
            process_image(img_file, model, label_dir, Path(output_dir), conf_threshold)
        print("全部处理完成！")