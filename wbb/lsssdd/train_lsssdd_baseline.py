# train_lsssdd_baseline.py — 标准 YOLO11s (yolo11s.yaml), LS-SSDD

from ultralytics import YOLO
import os, torch

MODEL_NAME = "yolo11s.yaml"
DATASET_PATH = "../cfg/ls_ssdd.yaml"
RUN_NAME = "lsssdd_baseline"

def setup_environment():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

def train_model():
    setup_environment()
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"使用 {'GPU: ' + torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'}")
    print(f"模型: {MODEL_NAME}  (run: {RUN_NAME})")
    model = YOLO(MODEL_NAME)
    try:
        results = model.train(
            data=DATASET_PATH, epochs=300, imgsz=640, batch=16, workers=0,
            device=device, optimizer="SGD", lr0=0.01, lrf=0.01,
            save=True, verbose=True, name=RUN_NAME,
        )
        print("训练完成!")
        return results
    except Exception as e:
        print(f"训练失败: {e}")
        return None

if __name__ == "__main__":
    train_model()
