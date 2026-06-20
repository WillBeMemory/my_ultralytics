import argparse
from ultralytics import YOLO

# 默认值
DEFAULT_MODEL = "./runs/detect/train/weights/best.pt"
DATA_YAML = "../cfg/hrsid_712rand.yaml"
IMGSZ = 640


def parse_args():
    p = argparse.ArgumentParser(description="HRSID(712rand)测试/评估脚本")
    p.add_argument("--run_name", type=str, required=True,
                   help="本次评估的 run 名(决定输出目录 runs/detect/<run_name>,便于区分不同模型/权重的评估结果)")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL,
                   help=f"权重 .pt 路径(默认:{DEFAULT_MODEL})")
    p.add_argument("--data", type=str, default=DATA_YAML,
                   help=f"数据集 yaml(默认:{DATA_YAML})")
    p.add_argument("--imgsz", type=int, default=IMGSZ,
                   help=f"输入尺寸(默认:{IMGSZ})")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = YOLO(args.model)
    results = model.val(data=args.data, split='test', imgsz=args.imgsz,
                        name=args.run_name, verbose=False)
    metrics = results.box

    def to_scalar(x):
        return x.item() if hasattr(x, 'item') else x

    print("\n========== Validation Results ==========")
    print(f"run_name:          {args.run_name}")
    print(f"model:             {args.model}")
    print(f"mAP@0.5:           {to_scalar(metrics.map50):.4f}")
    print(f"mAP@0.5:0.95:      {to_scalar(metrics.map):.4f}")
    print(f"Average Precision: {to_scalar(metrics.mp):.4f}")
    print(f"Average Recall:    {to_scalar(metrics.mr):.4f}")
    print(f"Average F1-score:  {to_scalar(metrics.f1):.4f}")
    print("=========================================")
