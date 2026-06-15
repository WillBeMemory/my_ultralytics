#!/usr/bin/env python3
"""
将 HRSID 数据集切分为 7:1:2 (train:val:test = 70%:10%:20%)。

服务器版：路径全部可配置(命令行参数 / 顶部 CONFIG)，核心逻辑与本地
wbb/make_hrsid_712_split.py 一致，固定 seed=42 —— 若服务器源数据与本地相同，
切出的 7:1:2 与本地完全一致，本地/服务器结果直接可比。

两种源结构：
  --mode twosplit (默认): 源为 {images/train, images/val}
                          test = 源 val(原样保留), val 从源 train 随机切出, train = 剩余
                          [与本地 HRSID_YOLO(4483/1121) 切法一致]
  --mode pool:            源为单一图池 {images/<pool_subdir>}
                          全部打散做全新 7:1:2 随机划分

实现：硬链接(hardlink, 零磁盘)；同卷失败回退复制。固定随机种子，可复现。

用法示例(服务器 Linux)：
  python3 make_hrsid_712_split_server.py \
      --src /data/HRSID_YOLO \
      --dst /data/HRSID_YOLO_712

  # 若服务器是单一图池(例如 images/all 里放全部图):
  python3 make_hrsid_712_split_server.py \
      --src /data/HRSID_official --mode pool --pool-subdir train
"""
import os
import argparse
import random
import shutil
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def img_to_lbl_name(img_name: str) -> str:
    return os.path.splitext(img_name)[0] + ".txt"


def link_or_copy(src: Path, dst: Path) -> str:
    """硬链接；同卷失败回退复制；目标已存在则跳过。"""
    if dst.exists():
        return "skip"
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def list_images(d: Path):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))


def emit(img_name, dst_split, src_root, src_split, dst_root):
    """把 src_root/images/<src_split>/<img> 及其标签放到 dst_root/.../<dst_split>/。"""
    img_src = src_root / "images" / src_split / img_name
    mode = link_or_copy(img_src, dst_root / "images" / dst_split / img_name)
    # 标签：src_root/labels/<src_split>/<name>.txt
    lbl_src = src_root / "labels" / src_split / img_to_lbl_name(img_name)
    lbl_dst = dst_root / "labels" / dst_split / img_to_lbl_name(img_name)
    if lbl_src.exists():
        link_or_copy(lbl_src, lbl_dst)
    else:
        lbl_dst.touch()  # 无目标 -> 空标签(YOLO 允许)
    return mode


def main():
    p = argparse.ArgumentParser(description="HRSID 7:1:2 splitter (server)")
    p.add_argument("--src", required=True, help="源数据集根目录(含 images/ 和 labels/)")
    p.add_argument("--dst", required=True, help="输出数据集根目录")
    p.add_argument("--mode", choices=["twosplit", "pool"], default="twosplit",
                   help="twosplit=源有 train/val(默认); pool=单一图池")
    p.add_argument("--pool-subdir", default="train",
                   help="pool 模式下图像所在子目录名(相对 images/), 默认 train")
    p.add_argument("--test-subdir", default="val",
                   help="twosplit 模式下作为 test 的源子目录名, 默认 val")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    assert src.exists(), f"源目录不存在: {src}"
    if dst.exists():
        print(f"⚠️ 目标已存在，按幂等模式继续(已存在文件跳过): {dst}")

    # 建目录
    for s in ("train", "val", "test"):
        (dst / "images" / s).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / s).mkdir(parents=True, exist_ok=True)

    cnt = {"link": 0, "copy": 0, "skip": 0}

    if args.mode == "twosplit":
        # 源: images/train + images/<test_subdir(=val)>
        src_train = list_images(src / "images" / "train")
        src_test = list_images(src / "images" / args.test_subdir)
        print(f"源 twosplit: train={len(src_train)}  {args.test_subdir}(->test)={len(src_test)}")

        # 从源 train 按 7:1 切出 val(1/8)
        rng = random.Random(args.seed)
        shuffled = src_train.copy()
        rng.shuffle(shuffled)
        n_val = round(len(shuffled) * (1 / 8))
        val_imgs = shuffled[:n_val]
        train_imgs = shuffled[n_val:]
        test_imgs = src_test[:]

        for f in train_imgs:
            cnt[emit(f, "train", src, "train", dst)] += 1
        for f in val_imgs:
            cnt[emit(f, "val", src, "train", dst)] += 1
        for f in test_imgs:
            cnt[emit(f, "test", src, args.test_subdir, dst)] += 1

        n_train, n_val, n_test = len(train_imgs), len(val_imgs), len(test_imgs)

    else:  # pool
        pool_imgs = list_images(src / "images" / args.pool_subdir)
        print(f"源 pool: {args.pool_subdir}={len(pool_imgs)}")
        rng = random.Random(args.seed)
        shuffled = pool_imgs.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = round(n * 0.2)
        n_val = round(n * 0.1)
        n_train = n - n_test - n_val
        train_imgs = shuffled[:n_train]
        val_imgs = shuffled[n_train:n_train + n_val]
        test_imgs = shuffled[n_train + n_val:]

        for f in train_imgs:
            cnt[emit(f, "train", src, args.pool_subdir, dst)] += 1
        for f in val_imgs:
            cnt[emit(f, "val", src, args.pool_subdir, dst)] += 1
        for f in test_imgs:
            cnt[emit(f, "test", src, args.pool_subdir, dst)] += 1

    tot = n_train + n_val + n_test
    print(f"\n新划分: train={n_train}  val={n_val}  test={n_test}  总计={tot}")
    print(f"比例: {n_train/tot*100:.1f}% / {n_val/tot*100:.1f}% / {n_test/tot*100:.1f}%")
    print(f"图像/标签处理方式: {cnt}")

    # 校验(含空标签计数 —— 梯度为0的常见根因)
    print("\n=== 校验 ===")
    all_ok = True
    for s in ("train", "val", "test"):
        ni = len(list_images(dst / "images" / s))
        labels = [f for f in os.listdir(dst / "labels" / s) if f.endswith(".txt")]
        nl = len(labels)
        empty = sum(1 for f in labels if (dst / "labels" / s / f).stat().st_size == 0)
        ok = (ni == nl)
        all_ok = all_ok and ok
        print(f"  {s}: images={ni}  labels={nl}  空标签={empty}  {'OK' if ok else 'MISMATCH!'}")

    # 生成 yaml
    yaml_path = dst.parent / (dst.name + ".yaml")
    yaml_text = (
        f"# auto-generated by make_hrsid_712_split_server.py\n"
        f"names:\n  0: ship\nnc: 1\n"
        f"path: {dst}\n"   # 服务器上用绝对路径最稳；如需相对路径自行修改
        f"train: images/train\nval: images/val\ntest: images/test\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"\n生成 yaml: {yaml_path}")
    print(f"\n{'✅ 完成' if all_ok else '⚠️ 存在 MISMATCH，请检查'}: {dst}")


if __name__ == "__main__":
    main()
