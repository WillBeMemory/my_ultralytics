"""
从现有 HRSID_YOLO (4483 train / 1121 val, 80/20) 派生 7:1:2 三划分。

设计：
  - test = 现有 1121 val（原样保留，保持与历史结果可比）
  - 从现有 4483 train 随机抽 560 -> 新 val
  - 剩余 3923 -> 新 train
  - 总计 3923 / 560 / 1121 = 70.0% / 10.0% / 20.0% (精确 7:1:2)

实现：硬链接(hardlink)创建，零磁盘占用；失败回退复制。固定随机种子，可复现。
"""
import os
import random
import shutil
from pathlib import Path

SRC = Path(r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_YOLO")
DST = Path(r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_YOLO_712")
SEED = 42
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def link_or_copy(src: Path, dst: Path):
    """硬链接；同卷失败则回退复制。"""
    if dst.exists():
        return "skip"
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def img_to_lbl_name(img_name: str) -> str:
    return os.path.splitext(img_name)[0] + ".txt"


def main():
    assert SRC.exists(), f"源目录不存在: {SRC}"
    if DST.exists():
        print(f"⚠️ 目标已存在，按幂等模式继续(已存在文件跳过): {DST}")

    # 现有数据
    old_train_imgs = sorted(f for f in os.listdir(SRC / "images" / "train") if f.lower().endswith(IMG_EXT))
    old_val_imgs = sorted(f for f in os.listdir(SRC / "images" / "val") if f.lower().endswith(IMG_EXT))
    print(f"现有: train={len(old_train_imgs)}  val={len(old_val_imgs)}")

    # 从 old_train 按 7:1 切出 val（1/8 -> 560）
    rng = random.Random(SEED)
    shuffled = old_train_imgs.copy()
    rng.shuffle(shuffled)
    n_val = round(len(shuffled) * (1 / 8))   # 4483/8 = 560.375 -> 560
    new_val_imgs = shuffled[:n_val]
    new_train_imgs = shuffled[n_val:]
    test_imgs = old_val_imgs[:]   # test = 原 val(1121) 原样

    print(f"新划分: train={len(new_train_imgs)}  val={len(new_val_imgs)}  test={len(test_imgs)}")
    tot = len(new_train_imgs) + len(new_val_imgs) + len(test_imgs)
    print(f"总计={tot}  比例={len(new_train_imgs)/tot*100:.1f}% / {len(new_val_imgs)/tot*100:.1f}% / {len(test_imgs)/tot*100:.1f}%")

    # 建目录
    for split in ("train", "val", "test"):
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)

    def emit(img_name, split):
        # 源 split: train/val 的新样本来自旧 train; test 来自旧 val
        src_split = "val" if split == "test" else "train"
        img_src = SRC / "images" / src_split / img_name
        mode = link_or_copy(img_src, DST / "images" / split / img_name)
        # 标签：SRC/labels/<src_split>/<name>.txt  (注意要回到数据集根，不是 images 下)
        lbl_src = SRC / "labels" / src_split / img_to_lbl_name(img_name)
        lbl_dst = DST / "labels" / split / img_to_lbl_name(img_name)
        if lbl_src.exists():
            link_or_copy(lbl_src, lbl_dst)
        else:
            lbl_dst.touch()  # 无目标 -> 空标签(YOLO 允许)
        return mode

    cnt = {"link": 0, "copy": 0, "skip": 0}
    for f in new_train_imgs:
        cnt[emit(f, "train")] += 1
    for f in new_val_imgs:
        cnt[emit(f, "val")] += 1
    for f in test_imgs:
        cnt[emit(f, "test")] += 1
    print(f"图像/标签处理方式: {cnt}")

    # 校验
    for split in ("train", "val", "test"):
        ni = len([f for f in os.listdir(DST / "images" / split) if f.lower().endswith(IMG_EXT)])
        labels = [f for f in os.listdir(DST / "labels" / split) if f.endswith(".txt")]
        nl = len(labels)
        empty = sum(1 for f in labels if (DST / "labels" / split / f).stat().st_size == 0)
        print(f"  校验 {split}: images={ni}  labels={nl}  空标签={empty}  {'OK' if ni == nl else 'MISMATCH!'}")

    print(f"\n完成: {DST}")


if __name__ == "__main__":
    main()
