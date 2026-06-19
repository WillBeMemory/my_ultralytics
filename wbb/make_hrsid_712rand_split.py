"""
完全随机 7:1:2 划分（pool 全部 5604，随机分 3923 / 560 / 1121）。

与 make_hrsid_712_split.py 的区别：
  - 旧 7:1:2：test = 原 80/20 val(1121) 固定不重分；train/val 从原 train 抽。
  - 本脚本：train/val/test 三者都从全 pool(5604) 完全随机抽 → 消除
    "test 固定/来源不均" 变量，用于验证 7:1:2 性能差是 "划分运气" 还是
    "train 量少(结构性)"。SEED 固定可复现。

跑法：python3 make_hrsid_712rand_split.py [SRC] [DST]  （默认本地路径）
"""
import os
import random
import shutil
import sys
from pathlib import Path

_SRC_DEFAULT = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_YOLO"
_DST_DEFAULT = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_YOLO_712rand"
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(_SRC_DEFAULT)
DST = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(_DST_DEFAULT)
SEED = 42
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return "skip"
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def img_to_lbl(name: str) -> str:
    return os.path.splitext(name)[0] + ".txt"


def main():
    assert SRC.exists(), f"源目录不存在: {SRC}"
    if DST.exists():
        print(f"⚠️ 目标已存在，幂等继续(已存在文件跳过): {DST}")

    pool = []
    for split in ("train", "val"):
        imgs = sorted(f for f in os.listdir(SRC / "images" / split) if f.lower().endswith(IMG_EXT))
        pool.extend((f, split) for f in imgs)
    print(f"pool 总数: {len(pool)}")

    rng = random.Random(SEED)
    rng.shuffle(pool)

    n = len(pool)
    n_train = round(n * 0.7)    # 3923
    n_val = round(n * 0.1)      # 560
    train_pool = pool[:n_train]
    val_pool = pool[n_train:n_train + n_val]
    test_pool = pool[n_train + n_val:]   # 1121

    tot = len(train_pool) + len(val_pool) + len(test_pool)
    print(f"划分: train={len(train_pool)} val={len(val_pool)} test={len(test_pool)} 总={tot}")
    print(f"比例={len(train_pool)/tot*100:.1f}% / {len(val_pool)/tot*100:.1f}% / {len(test_pool)/tot*100:.1f}%")

    for split in ("train", "val", "test"):
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)

    def emit(img_name, src_split, split):
        img_src = SRC / "images" / src_split / img_name
        mode = link_or_copy(img_src, DST / "images" / split / img_name)
        lbl_src = SRC / "labels" / src_split / img_to_lbl(img_name)
        lbl_dst = DST / "labels" / split / img_to_lbl(img_name)
        if lbl_src.exists():
            link_or_copy(lbl_src, lbl_dst)
        else:
            lbl_dst.touch()
        return mode

    cnt = {"link": 0, "copy": 0, "skip": 0}
    for f, s in train_pool:
        cnt[emit(f, s, "train")] += 1
    for f, s in val_pool:
        cnt[emit(f, s, "val")] += 1
    for f, s in test_pool:
        cnt[emit(f, s, "test")] += 1
    print(f"图像/标签处理方式: {cnt}")

    names = {}
    for split in ("train", "val", "test"):
        ni = len([f for f in os.listdir(DST / "images" / split) if f.lower().endswith(IMG_EXT)])
        labels = [f for f in os.listdir(DST / "labels" / split) if f.endswith(".txt")]
        nl = len(labels)
        empty = sum(1 for f in labels if (DST / "labels" / split / f).stat().st_size == 0)
        print(f"  校验 {split}: images={ni} labels={nl} 空标签={empty} {'OK' if ni == nl else 'MISMATCH!'}")
        names[split] = set(f for f in os.listdir(DST / "images" / split) if f.lower().endswith(IMG_EXT))

    tv = len(names["train"] & names["val"])
    tt = len(names["train"] & names["test"])
    vt = len(names["val"] & names["test"])
    print(f"  两两互斥: train∩val={tv} train∩test={tt} val∩test={vt} (应全 0)")

    print(f"\n完成: {DST}")


if __name__ == "__main__":
    main()
