import argparse
import os
from ultralytics import YOLO

DATA_YAML = "../cfg/ssdd_official.yaml"
IMGSZ = 640
RUNS_DIR = "./runs/detect"


def parse_args():
    p = argparse.ArgumentParser(description="SSDD(official) WEF-YOLO 测试/评估脚本")
    p.add_argument("--run_name", type=str, required=True,
                   help="训练时的 run 名;权重自动取 ./runs/detect/<run_name>/weights/best.pt")
    p.add_argument("--data", type=str, default=DATA_YAML,
                   help=f"数据集 yaml(默认:{DATA_YAML})")
    p.add_argument("--imgsz", type=int, default=IMGSZ,
                   help=f"输入尺寸(默认:{IMGSZ})")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model_path = os.path.join(RUNS_DIR, args.run_name, "weights", "best.pt")
    print(f"权重路径: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到权重文件 {model_path},请确认 --run_name 与训练时的 name 一致。")

    model = YOLO(model_path)
    results = model.val(data=args.data, split='test', imgsz=args.imgsz,
                        name=args.run_name, verbose=False)
    metrics = results.box

    def to_scalar(x):
        return x.item() if hasattr(x, 'item') else x

    print("\n========== Validation Results ==========")
    print(f"run_name:          {args.run_name}")
    print(f"model:             {model_path}")
    print(f"mAP@0.5:           {to_scalar(metrics.map50):.4f}")
    print(f"mAP@0.5:0.95:      {to_scalar(metrics.map):.4f}")
    print(f"Average Precision: {to_scalar(metrics.mp):.4f}")
    print(f"Average Recall:    {to_scalar(metrics.mr):.4f}")
    print(f"Average F1-score:  {to_scalar(metrics.f1):.4f}")
    print("=========================================")
