"""
将官方 SSDD (COCO格式) 转换为 YOLO 格式，保留官方 train/test 划分。
使用本地 SSDD 的图片 + 官方 JSON 标注。
输出目录: SSDD_official
"""
import json
import shutil
import os
from pathlib import Path

# 官方 COCO 标注路径（已解压）
ANN_DIR = Path(r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\Official-SSDD-OPEN\Official-SSDD-OPEN\BBox_SSDD\coco_style\annotations")

# 本地 SSDD 图片（所有图片都在这里，按编号查找）
LOCAL_IMGS = Path(r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\SSDD")

# 输出目录
DST = Path(r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\SSDD_official")

# 类别映射
CATEGORIES = {0: 0}  # SSDD official uses category_id=0


def convert_bbox_coco_to_yolo(img_w, img_h, bbox):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return [x_center, y_center, w_norm, h_norm]


def find_local_image(stem):
    """在本地 SSDD 的 train/val/test 中查找图片"""
    for split in ["train", "val", "test"]:
        path = LOCAL_IMGS / "images" / split / f"{stem}.jpg"
        if path.exists():
            return path
    return None


def process_split(json_name, split_name):
    json_path = ANN_DIR / json_name
    with open(json_path, "r") as f:
        data = json.load(f)

    img_dict = {img["id"]: img for img in data["images"]}

    img_anns = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)

    img_dst = DST / "images" / split_name
    lbl_dst = DST / "labels" / split_name
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    count = 0
    missing = 0
    for img_id, img_info in img_dict.items():
        file_name = img_info["file_name"]  # e.g., "000001.jpg"
        stem = Path(file_name).stem
        img_w = img_info["width"]
        img_h = img_info["height"]

        # 从本地 SSDD 查找图片
        src_img = find_local_image(stem)
        if src_img is None:
            missing += 1
            continue

        # 复制图片
        dst_img = img_dst / file_name
        shutil.copy2(src_img, dst_img)

        # 生成 YOLO 标签
        label_name = stem + ".txt"
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

    print(f"  {split_name}: {count} images, {missing} missing")
    return count


if __name__ == "__main__":
    print("Converting Official SSDD (COCO) -> SSDD_official (YOLO)")
    print(f"Annotations: {ANN_DIR}")
    print(f"Local images: {LOCAL_IMGS}")
    print(f"Dest: {DST}")
    print()

    train_count = process_split("train.json", "train")
    test_count = process_split("test.json", "test")

    print()
    print(f"Done! Total: {train_count + test_count} images")
    print(f"  train: {train_count}")
    print(f"  test:  {test_count}")

    # 生成 yaml
    yaml_content = """names:
  0: ship
nc: 1
path: ../datasets/SSDD_official
train: images/train
val: images/test
test: images/test
"""
    yaml_path = DST.parent.parent / "cfg" / "ssdd_official.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\nYAML config saved to: {yaml_path}")
