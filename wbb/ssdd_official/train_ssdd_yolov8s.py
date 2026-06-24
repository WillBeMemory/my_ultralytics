# train_ssdd_yolov8s.py — YOLOv8s baseline, SSDD 官方划分

from ultralytics import YOLO
import os, torch

MODEL_NAME = "yolov8s.yaml"
DATASET_PATH = "../cfg/ssdd_official.yaml"
RUN_NAME = "ssdd_yolov8s"

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
            data=DATASET_PATH, epochs=300, imgsz=512, batch=16, workers=0,
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
