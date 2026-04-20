import os
import cv2
import numpy as np

def draw_yolo_annotations(image_path, label_path, class_names=None, output_path=None, show_image=True):
    """
    在图片上绘制YOLO格式的标注，并输出标注信息到控制台。

    参数：
        image_path: 图片文件路径
        label_path: 对应的YOLO标注txt文件路径
        class_names: 类别名称列表（索引对应类别ID），可选，若提供则在框上显示名称
        output_path: 保存绘制后图片的路径，若为None则不保存
        show_image: 是否显示图片（按任意键关闭）
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图片 {image_path}")
        return
    height, width = img.shape[:2]

    # 读取标注文件
    if not os.path.isfile(label_path):
        print(f"警告：标注文件不存在 {label_path}")
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()

    print(f"\n图片: {os.path.basename(image_path)}")
    print(f"尺寸: {width} x {height}")
    print("检测到的目标:")

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            print(f"  格式错误: {line}")
            continue

        class_id = int(parts[0])
        x_center_norm = float(parts[1])
        y_center_norm = float(parts[2])
        w_norm = float(parts[3])
        h_norm = float(parts[4])

        # 反归一化到像素坐标
        x_center = x_center_norm * width
        y_center = y_center_norm * height
        box_w = w_norm * width
        box_h = h_norm * height

        # 计算左上角和右下角坐标
        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)

        # 确保坐标在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        # 获取类别名称
        class_name = class_names[class_id] if class_names and class_id < len(class_names) else str(class_id)

        # 控制台输出
        print(f"  类别: {class_name} (ID {class_id})")
        print(f"    归一化中心: ({x_center_norm:.4f}, {y_center_norm:.4f}) 尺寸: ({w_norm:.4f}, {h_norm:.4f})")
        print(f"    像素坐标: 左上({x1}, {y1}) 右下({x2}, {y2}) 尺寸: ({box_w:.1f}, {box_h:.1f})")

        # 绘制矩形
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制标签背景
        label = class_name
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - label_h - baseline), (x1 + label_w, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 保存图片
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        print(f"已保存标注图片至: {output_path}")

    # 显示图片
    if show_image:
        cv2.imshow('YOLO Annotations', img)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def batch_process(image_folder, label_folder, class_names=None, output_folder=None, show_images=False):
    """
    批量处理文件夹中的图片和标注。
    假设图片和标注文件名相同（仅扩展名不同），图片扩展名支持常见格式。
    """
    # 支持的图片扩展名
    image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(image_extensions):
            base = os.path.splitext(filename)[0]
            image_path = os.path.join(image_folder, filename)
            label_path = os.path.join(label_folder, base + '.txt')

            if not os.path.isfile(label_path):
                print(f"跳过 {filename}：无对应标注文件")
                continue

            if output_folder:
                output_path = os.path.join(output_folder, f"annotated_{filename}")
            else:
                output_path = None

            draw_yolo_annotations(image_path, label_path, class_names, output_path, show_image=show_images)


if __name__ == "__main__":
    # ==================== 配置区域 ====================
    # 单张图片处理模式（注释掉批量模式即可）
    # image_path = "path/to/image.jpg"
    # label_path = "path/to/image.txt"
    # class_names_file = "path/to/classes.txt"   # 可选，每行一个类别名
    # output_path = "path/to/output.jpg"          # 可选，保存绘制后的图片
    # show_image = True

    # 批量处理模式（处理整个文件夹）
    image_folder = r"D:\Study\10_dataset\HRSID_YOLO_COMPLEX\images\all"          # 图片文件夹
    label_folder = r"D:\Study\10_dataset\HRSID_YOLO_COMPLEX\labels\all"          # 标注文件夹（txt文件）
    class_names_file = None # 可选，类别名称文件
    output_folder = r"D:\Study\10_dataset\HRSID_YOLO_COMPLEX\draw"         # 可选，保存标注后图片的文件夹
    show_images = False                       # 是否显示图片（批量处理时建议False，避免弹出大量窗口）
    # ==================================================

    # 加载类别名称（如果提供了文件）
    class_names = None
    if class_names_file and os.path.isfile(class_names_file):
        with open(class_names_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]

    # 选择单张或批量处理
    # 单张处理（取消下面注释，并注释掉批量处理）
    # draw_yolo_annotations(image_path, label_path, class_names, output_path, show_image)

    # 批量处理
    batch_process(image_folder, label_folder, class_names, output_folder, show_images)