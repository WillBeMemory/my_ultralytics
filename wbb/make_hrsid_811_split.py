"""
从现有 HRSID_YOLO (4483 train / 1121 val, 80/20) 派生 8:1:1 三划分。

设计：
  - pool 全部 5604 张(old train 4483 + old val 1121)，固定种子重洗
  - 切 8:1:1 -> train 4483 / val 560 / test 561（两两互斥，val != test）
  - train 与 80/20 同规模(4483) -> 绝对数字预期接近 80/20，保 SOTA 叙事
  - val/test 独立 -> 解决 80/20 的 val=test 方法學硬伤

实现：硬链接(hardlink)零磁盘；失败回退复制。固定 SEED 可复现。
⚠️ 注意：test 集从全 pool 随机抽(561)，与 80/20 的 1121 / 7:1:2 的 1121 都不同，
   => 所有 HRSID 结果须在 8:1:1 上重训 baseline+Ours，历史数字不可直接搬。
"""
import os
import random
import shutil
import sys
from pathlib import Path

# 默认本地 Windows 路径；服务器上用命令行参数覆盖：
#   python3 make_hrsid_811_split.py <SRC(80/20 源)> <DST(8:1:1 输出)>
_SRC_DEFAULT = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_YOLO"
_DST_DEFAULT = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_YOLO_811"
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(_SRC_DEFAULT)
DST = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(_DST_DEFAULT)
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


def img_to_lbl(name: str) -> str:
    return os.path.splitext(name)[0] + ".txt"


def main():
    assert SRC.exists(), f"源目录不存在: {SRC}"
    if DST.exists():
        print(f"⚠️ 目标已存在，幂等继续(已存在文件跳过): {DST}")

    # pool 全部图像，记录每张的来源 split(old train / old val)，便于回溯找 label
    pool = []
    for split in ("train", "val"):
        imgs = sorted(f for f in os.listdir(SRC / "images" / split) if f.lower().endswith(IMG_EXT))
        pool.extend((f, split) for f in imgs)
    print(f"pool 总数: {len(pool)}")

    rng = random.Random(SEED)
    rng.shuffle(pool)

    n = len(pool)
    n_train = round(n * 0.8)    # 4483
    n_val = round(n * 0.1)      # 560
    train_pool = pool[:n_train]
    val_pool = pool[n_train:n_train + n_val]
    test_pool = pool[n_train + n_val:]   # 剩余 -> 561

    tot = len(train_pool) + len(val_pool) + len(test_pool)
    print(f"新划分: train={len(train_pool)} val={len(val_pool)} test={len(test_pool)} 总计={tot}")
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
            lbl_dst.touch()  # 无目标 -> 空标签(YOLO 允许)
        return mode

    cnt = {"link": 0, "copy": 0, "skip": 0}
    for f, s in train_pool:
        cnt[emit(f, s, "train")] += 1
    for f, s in val_pool:
        cnt[emit(f, s, "val")] += 1
    for f, s in test_pool:
        cnt[emit(f, s, "test")] += 1
    print(f"图像/标签处理方式: {cnt}")

    # 校验：数量、空标签、两两互斥
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
