"""SSDD eval script with detailed metrics: AP50/AP70/AP95, APS/APM/APL.

Usage:
    cd wbb/ssdd_official
    python eval_ssdd_WEF.py
    python eval_ssdd_WEF.py --run_name train2
"""
import argparse
import json
import os
import numpy as np
from pathlib import Path
from ultralytics import YOLO

DATA_YAML = "../cfg/ssdd_official.yaml"
IMGSZ = 512
RUNS_DIR = "./runs/detect"

# COCO area thresholds (pixels^2 in original image)
AREA_SMALL_MAX = 32 ** 2     # 1024
AREA_MEDIUM_MAX = 96 ** 2    # 9216


def find_latest_run(runs_dir):
    """Find the latest training run with best.pt."""
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return None
    candidates = []
    for d in runs_path.iterdir():
        if d.is_dir() and (d / "weights" / "best.pt").exists():
            candidates.append(d)
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x.stat().st_mtime)[-1].name


def parse_args():
    p = argparse.ArgumentParser(description="SSDD detailed eval script")
    default_run = find_latest_run(RUNS_DIR)
    p.add_argument("--run_name", type=str, default=default_run,
                   help=f"Training run name (default: latest run = {default_run})")
    p.add_argument("--data", type=str, default=DATA_YAML)
    p.add_argument("--imgsz", type=int, default=IMGSZ)
    p.add_argument("--split", type=str, default="test")
    return p.parse_args()


def yolo_labels_to_coco(data_path, split, imgsz, use_stem_id=False):
    """Convert YOLO labels to COCO-format JSON for pycocotools evaluation.

    Args:
        data_path: dataset root directory
        split: dataset split name (e.g. 'test')
        imgsz: image size (used as fallback if image can't be opened)
        use_stem_id: if True, use file stem as image_id (matching ultralytics save_json output)
    """
    from PIL import Image

    data_path = Path(data_path)
    img_dir = data_path / "images" / split
    label_dir = data_path / "labels" / split

    if not label_dir.exists():
        label_dir = data_path / "labels" / "test"

    images = []
    annotations = []
    ann_id = 1

    for img_id_counter, img_file in enumerate(sorted(img_dir.glob("*")), start=1):
        stem = img_file.stem
        label_file = label_dir / f"{stem}.txt"

        # Get image size
        try:
            with Image.open(img_file) as im:
                w_orig, h_orig = im.size
        except Exception:
            w_orig, h_orig = imgsz, imgsz

        # image_id: match ultralytics convention
        # ultralytics: image_id = int(stem) if stem.isnumeric() else stem
        if use_stem_id:
            image_id = int(stem) if stem.isnumeric() else stem
        else:
            image_id = img_id_counter

        images.append({
            "id": image_id,
            "file_name": img_file.name,
            "width": w_orig,
            "height": h_orig,
        })

        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                    # Convert normalized xywh to COCO xywh (absolute pixels)
                    abs_w = bw * w_orig
                    abs_h = bh * h_orig
                    abs_x = cx * w_orig - abs_w / 2
                    abs_y = cy * h_orig - abs_h / 2

                    annotations.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": cls_id + 1,  # COCO is 1-indexed
                        "bbox": [abs_x, abs_y, abs_w, abs_h],
                        "area": abs_w * abs_h,
                        "iscrowd": 0,
                    })
                    ann_id += 1

    categories = [{"id": i + 1, "name": f"class_{i}"} for i in range(1)]  # nc=1

    return {"images": images, "annotations": annotations, "categories": categories}


def main():
    args = parse_args()
    model_path = os.path.join(RUNS_DIR, args.run_name, "weights", "best.pt")
    print(f"Model: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Weight not found: {model_path}")

    model = YOLO(model_path)

    # ---- Standard validation ----
    results = model.val(data=args.data, split=args.split, imgsz=args.imgsz, verbose=False)
    m = results.box

    def s(x):
        return x.item() if hasattr(x, 'item') else float(x)

    # all_ap: shape (nc, 10), IoU thresholds: 0.5, 0.55, 0.6, ..., 0.95
    all_ap = m.all_ap  # (nc, 10)

    iou_names = ["0.50", "0.55", "0.60", "0.65", "0.70", "0.75", "0.80", "0.85", "0.90", "0.95"]
    ap_per_iou = {}
    for i, name in enumerate(iou_names):
        ap_per_iou[name] = all_ap[:, i].mean() if len(all_ap) else 0.0

    # ---- APS/APM/APL via pycocotools ----
    aps = apm = apl = -1.0
    try:
        import yaml
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        with open(args.data, 'r', encoding='utf-8') as f:
            data_cfg = yaml.safe_load(f)

        data_path = Path(data_cfg['path'])
        if not data_path.is_absolute():
            data_path = Path(args.data).parent / data_path
        data_path = data_path.resolve()

        # Step 1: Build GT JSON with sequential integer image_ids
        print("Generating COCO GT annotations...")
        coco_gt_dict = yolo_labels_to_coco(str(data_path), args.split, args.imgsz, use_stem_id=False)

        # Build filename -> integer_id mapping from GT
        fname_to_id = {img['file_name']: img['id'] for img in coco_gt_dict['images']}

        gt_json_path = data_path / f"_coco_gt_{args.split}.json"
        with open(gt_json_path, 'w') as f:
            json.dump(coco_gt_dict, f)

        # Step 2: Run val with save_json to get COCO-format predictions
        coco_eval_name = f"_coco_eval_{args.run_name}"
        print("Running val with save_json for COCO eval...")
        val_results = model.val(data=args.data, split=args.split, imgsz=args.imgsz,
                                save_json=True, verbose=False, name=coco_eval_name)

        # Step 3: Find the saved predictions JSON
        pred_json = None
        val_dir = Path(RUNS_DIR) / coco_eval_name
        if val_dir.exists():
            candidates = list(val_dir.rglob("*.json"))
            for c in sorted(candidates):
                try:
                    with open(c, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, list) and len(data) > 0 and "image_id" in data[0]:
                        pred_json = c
                        break
                except Exception:
                    continue

        if pred_json and pred_json.exists():
            with open(pred_json, 'r') as f:
                pred_data = json.load(f)

            # category_id: ultralytics non-COCO datasets already use 1-indexed (class_map = range(1, nc+1))
            # No need to add 1 again

            # Remap image_id: from stem string to integer matching GT
            for item in pred_data:
                fname = item.get('file_name', '')
                if fname in fname_to_id:
                    item['image_id'] = fname_to_id[fname]
                else:
                    stem = item.get('image_id', '')
                    for fn, iid in fname_to_id.items():
                        if Path(fn).stem == stem:
                            item['image_id'] = iid
                            break

            fixed_pred_path = data_path / f"_coco_pred_{args.split}.json"
            with open(fixed_pred_path, 'w') as f:
                json.dump(pred_data, f)

            coco_gt = COCO(str(gt_json_path))
            coco_dt = coco_gt.loadRes(str(fixed_pred_path))

            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            # COCO stats: [AP, AP50, AP75, APs, APm, APl, AR1, AR10, ARs, ARm, ARl]
            aps = coco_eval.stats[3]
            apm = coco_eval.stats[4]
            apl = coco_eval.stats[5]

            # Clean up temp files
            fixed_pred_path.unlink(missing_ok=True)
            gt_json_path.unlink(missing_ok=True)
        else:
            print("[Warning] COCO predictions JSON not found, skipping APS/APM/APL")
            gt_json_path.unlink(missing_ok=True)

    except ImportError:
        print("[Warning] pycocotools not installed, skipping APS/APM/APL")
        print("         Install with: pip install pycocotools")
    except Exception as e:
        print(f"[Warning] COCO eval failed: {e}")

    # ---- Print results ----
    print("\n" + "=" * 55)
    print(f"  Run: {args.run_name}")
    print(f"  Model: {model_path}")
    print("=" * 55)

    print("\n--- mAP Summary ---")
    print(f"  mAP@0.5:      {s(m.map50):.4f}")
    print(f"  mAP@0.5:0.95: {s(m.map):.4f}")
    print(f"  mAP@0.75:     {s(m.map75):.4f}")

    print("\n--- AP at specific IoU thresholds ---")
    for name, val in ap_per_iou.items():
        print(f"  AP@{name}: {val:.4f}")

    print("\n--- AP by object size (COCO definition) ---")
    if aps >= 0:
        print(f"  AP_S (area < {AREA_SMALL_MAX}):       {aps:.4f}")
        print(f"  AP_M ({AREA_SMALL_MAX}<=area<{AREA_MEDIUM_MAX}): {apm:.4f}")
        print(f"  AP_L (area >= {AREA_MEDIUM_MAX}):     {apl:.4f}")
    else:
        print("  (Not available - pycocotools required)")

    print("\n--- Other metrics ---")
    print(f"  Precision: {s(m.mp):.4f}")
    print(f"  Recall:    {s(m.mr):.4f}")
    print(f"  F1-score:  {s(m.f1):.4f}")
    print("=" * 55)


if __name__ == '__main__':
    main()
