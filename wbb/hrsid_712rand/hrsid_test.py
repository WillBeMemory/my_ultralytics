from ultralytics import YOLO

model_path = "./runs/detect/train/weights/best.pt"
data_yaml = "../cfg/hrsid_712rand.yaml"
imgsz = 640

if __name__ == '__main__':
    model = YOLO(model_path)
    results = model.val(data=data_yaml, split='test', imgsz=imgsz, verbose=False)
    metrics = results.box

    def to_scalar(x):
        return x.item() if hasattr(x, 'item') else x

    print("\n========== Validation Results ==========")
    print(f"mAP@0.5:          {to_scalar(metrics.map50):.4f}")
    print(f"mAP@0.5:0.95:     {to_scalar(metrics.map):.4f}")
    print(f"Average Precision: {to_scalar(metrics.mp):.4f}")
    print(f"Average Recall:    {to_scalar(metrics.mr):.4f}")
    print(f"Average F1-score:  {to_scalar(metrics.f1):.4f}")
    print("=========================================")
