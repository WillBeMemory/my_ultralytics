import argparse, os
from ultralytics import YOLO

DATA_YAML = "../cfg/ls_ssdd.yaml"
IMGSZ = 640
RUNS_DIR = "./runs/detect"

def parse_args():
    p = argparse.ArgumentParser(description="LS-SSDD WEF-YOLO 测试/评估脚本")
    p.add_argument("--run_name", type=str, required=True,
                   help="训练时的 run 名;权重自动取 ./runs/detect/<run_name>/weights/best.pt")
    p.add_argument("--data", type=str, default=DATA_YAML)
    p.add_argument("--imgsz", type=int, default=IMGSZ)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model_path = os.path.join(RUNS_DIR, args.run_name, "weights", "best.pt")
    print(f"权重路径: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到权重 {model_path}")
    model = YOLO(model_path)
    results = model.val(data=args.data, split='test', imgsz=args.imgsz,
                        name=args.run_name, verbose=False)
    metrics = results.box
    def s(x): return x.item() if hasattr(x, 'item') else x
    print("\n========== Validation Results ==========")
    print(f"run_name:          {args.run_name}")
    print(f"mAP@0.5:           {s(metrics.map50):.4f}")
    print(f"mAP@0.5:0.95:      {s(metrics.map):.4f}")
    print(f"Average Precision: {s(metrics.mp):.4f}")
    print(f"Average Recall:    {s(metrics.mr):.4f}")
    print(f"Average F1-score:  {s(metrics.f1):.4f}")
    print("=========================================")
