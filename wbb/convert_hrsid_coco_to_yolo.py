"""
将 HRSID_JPG (COCO格式) 转换为 YOLO 格式，保留官方 train/test 划分。
输出目录: HRSID_YOLO_official
"""

import json
import shutil
from pathlib import Path

# 路径配置
SRC = Path(r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_JPG")
DST = Path(r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_YOLO_official")

# 类别映射 (HRSID 只有 ship 一类)
CATEGORIES = {1: 0}  # COCO category_id 1 -> YOLO class 0


def convert_bbox_coco_to_yolo(img_w, img_h, bbox):
    """COCO [x, y, w, h] -> YOLO [x_center, y_center, w, h] (归一化)"""
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return [x_center, y_center, w_norm, h_norm]


def process_split(json_name, split_name):
    """处理一个划分 (train 或 test)"""
    json_path = SRC / "annotations" / json_name
    with open(json_path, "r") as f:
        data = json.load(f)

    # 构建 image_id -> image_info 映射
    img_dict = {img["id"]: img for img in data["images"]}

    # 构建 image_id -> annotations 映射
    img_anns = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)

    # 创建目标目录
    img_dst = DST / "images" / split_name
    lbl_dst = DST / "labels" / split_name
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_id, img_info in img_dict.items():
        file_name = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        # 复制图片
        src_img = SRC / "JPEGImages" / file_name
        dst_img = img_dst / file_name
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
        else:
            print(f"  Warning: {src_img} not found")
            continue

        # 生成 YOLO 标签
        label_name = Path(file_name).stem + ".txt"
        dst_lbl = lbl_dst / label_name

        anns = img_anns.get(img_id, [])
        with open(dst_lbl, "w") as f:
            for ann in anns:
                cat_id = ann["category_id"]
                if cat_id not in CATEGORIES:
                    continue
                yolo_cls = CATEGORIES[cat_id]
                bbox = ann["bbox"]
                yolo_bbox = convert_bbox_coco_to_yolo(img_w, img_h, bbox)
                f.write(f"{yolo_cls} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")

        count += 1

    print(f"  {split_name}: {count} images, {sum(len(v) for v in img_anns.values())} annotations")
    return count


if __name__ == "__main__":
    print("Converting HRSID_JPG (COCO) -> HRSID_YOLO_official (YOLO)")
    print(f"Source: {SRC}")
    print(f"Dest:   {DST}")
    print()

    train_count = process_split("train2017.json", "train")
    test_count = process_split("test2017.json", "test")

    print()
    print(f"Done! Total: {train_count + test_count} images")
    print(f"  train: {train_count}")
    print(f"  test:  {test_count}")

    # 生成 yaml 配置文件
    yaml_content = f"""names:
  0: ship
nc: 1
path: ../datasets/HRSID_YOLO_official
train: images/train
val: images/test
test: images/test
"""
    yaml_path = DST.parent.parent / "cfg" / "hrsid_official.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\nYAML config saved to: {yaml_path}")
