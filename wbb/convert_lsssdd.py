"""LS-SSDD VOC XML → YOLO 格式转换（纯标准库，无需 pip install）

用法：在 datasets/ 目录下（ls-ssdd_raw/ 旁边）运行：
    python convert_lsssdd.py

输入结构（Kaggle 下载解压后）：
    ls-ssdd_raw/
      Annotations_sub/Annotations_sub/*.xml   ← VOC 标注
      JPEGImages_sub_train/*.jpg              ← 训练图像
      JPEGImages_sub_test/*.jpg               ← 测试图像

输出结构（与 ls_ssdd.yaml 一致）：
    LS_SSDD_YOLO/
      images/train/  ← symlink 到原图
      images/test/
      labels/train/  ← YOLO txt
      labels/test/
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path

SRC = Path("ls-ssdd_raw")
DST = Path("LS_SSDD_YOLO")
ANNOT = SRC / "Annotations_sub" / "Annotations_sub"


def voc_to_yolo(xml_path, img_w=800, img_h=800):
    """将单个 VOC XML 转为 YOLO txt 行列表。无标注则返回空列表（背景图）。"""
    lines = []
    if not xml_path.exists():
        return lines
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is not None:
        w = int(size.find("width").text)
        h = int(size.find("height").text)
    else:
        w, h = img_w, img_h
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        xmin = float(b.find("xmin").text)
        ymin = float(b.find("ymin").text)
        xmax = float(b.find("xmax").text)
        ymax = float(b.find("ymax").text)
        xc = (xmin + xmax) / 2.0 / w
        yc = (ymin + ymax) / 2.0 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def process(img_subdir, split):
    # Kaggle 解压后 JPEGImages_sub_train/JPEGImages_sub_train/ 有嵌套,自动适配
    nested = img_subdir / img_subdir.name
    img_dir = nested if nested.is_dir() else img_subdir
    dst_img = DST / "images" / split
    dst_lbl = DST / "labels" / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    imgs = sorted(f for f in os.listdir(img_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png")))
    n_empty = 0
    for name in imgs:
        base = os.path.splitext(name)[0]
        # 转换标注
        lines = voc_to_yolo(ANNOT / f"{base}.xml")
        if not lines:
            n_empty += 1
        (dst_lbl / f"{base}.txt").write_text("\n".join(lines))
        # symlink 图像（省磁盘）
        dst = dst_img / name
        if not dst.exists():
            os.symlink((img_dir / name).resolve(), dst)
    print(f"  {split}: {len(imgs)} images, {n_empty} empty labels")
    return len(imgs), n_empty


if __name__ == "__main__":
    print("=== LS-SSDD VOC → YOLO 转换 ===")
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "images").mkdir(exist_ok=True)
    (DST / "labels").mkdir(exist_ok=True)

    process(SRC / "JPEGImages_sub_train", "train")
    process(SRC / "JPEGImages_sub_test", "test")

    # 验证
    print("\n=== 验证 ===")
    for split in ("train", "test"):
        ni = len(os.listdir(DST / "images" / split))
        nl = len(os.listdir(DST / "labels" / split))
        print(f"  {split}: {ni} images, {nl} labels", "OK" if ni == nl else "MISMATCH!")
    print("\nDone! LS_SSDD_YOLO/ 已就绪,可直接用于训练。")
