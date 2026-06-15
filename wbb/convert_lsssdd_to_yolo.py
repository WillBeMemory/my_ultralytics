"""
LS-SSDD-v1.0-OPEN (VOC格式) → YOLO格式转换
使用官方划分: train = 01-10, test = 11-15
"""
import os
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

# 路径配置
BASE = Path(r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\LS-SSDD-v1.0-OPEN")
OUTPUT = Path(r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\LS_SSDD_YOLO")

ANNO_DIR = BASE / "Annotations_sub"
IMG_DIR = BASE / "JPEGImages_sub"
SPLIT_DIR = BASE / "ImageSets" / "Main"

# 类别映射
CLASSES = {"ship": 0}

def convert_voc_to_yolo(xml_path, img_w, img_h):
    """将VOC XML转换为YOLO格式的标注列表"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    yolo_labels = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        if cls_name not in CLASSES:
            continue
        cls_id = CLASSES[cls_name]
        
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        
        # 转换为YOLO格式 (归一化中心点+宽高)
        x_center = (xmin + xmax) / 2.0 / img_w
        y_center = (ymin + ymax) / 2.0 / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h
        
        yolo_labels.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_labels


def process_split(split_name, file_list):
    """处理一个划分（train/test）"""
    img_out = OUTPUT / "images" / split_name
    lbl_out = OUTPUT / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    
    count = 0
    has_obj = 0
    total_obj = 0
    
    for image_id in file_list:
        xml_path = ANNO_DIR / f"{image_id}.xml"
        img_path = IMG_DIR / f"{image_id}.jpg"
        
        if not xml_path.exists():
            print(f"  [警告] 缺少标注: {image_id}")
            continue
        if not img_path.exists():
            print(f"  [警告] 缺少图片: {image_id}")
            continue
        
        # 转换标注
        yolo_labels = convert_voc_to_yolo(xml_path, 800, 800)
        
        # 复制图片
        shutil.copy2(img_path, img_out / f"{image_id}.jpg")
        
        # 写标签文件
        with open(lbl_out / f"{image_id}.txt", "w") as f:
            f.write("\n".join(yolo_labels) + "\n" if yolo_labels else "")
        
        count += 1
        if yolo_labels:
            has_obj += 1
            total_obj += len(yolo_labels)
    
    print(f"  {split_name}: {count} 张图片, {has_obj} 张有目标, 共 {total_obj} 个目标")


# 主流程
print("=" * 60)
print("LS-SSDD-v1.0-OPEN → YOLO 格式转换")
print("=" * 60)

# 读取官方划分
train_ids = []
with open(SPLIT_DIR / "train.txt", "r") as f:
    train_ids = [line.strip() for line in f if line.strip()]

test_ids = []
with open(SPLIT_DIR / "test.txt", "r") as f:
    test_ids = [line.strip() for line in f if line.strip()]

print(f"\n官方划分:")
print(f"  Train: {len(train_ids)} 张 (原始图像 01-10)")
print(f"  Test:  {len(test_ids)} 张 (原始图像 11-15)")
print(f"  Total: {len(train_ids) + len(test_ids)} 张\n")

# 处理
process_split("train", train_ids)
process_split("test", test_ids)

# 创建 data.yaml
yaml_path = OUTPUT / "ls_ssdd.yaml"
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(f"""# LS-SSDD-v1.0-OPEN YOLO format (官方划分)
# Train: 原始图像 01-10, Test: 原始图像 11-15
path: {OUTPUT.as_posix()}
train: images/train
val: images/test
test: images/test

nc: 1
names: ['ship']
""")

print(f"\n完成!")
print(f"输出目录: {OUTPUT}")
print(f"配置文件: {yaml_path}")
print(f"图片: {OUTPUT / 'images' / 'train'} ({len(train_ids)} 张)")
print(f"图片: {OUTPUT / 'images' / 'test'} ({len(test_ids)} 张)")