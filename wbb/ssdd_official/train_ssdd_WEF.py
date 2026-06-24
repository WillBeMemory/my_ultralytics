# train_ssdd_WEF.py — WEF-YOLO (yolo11s-ablation-WEF.yaml), SSDD 官方划分
# 数据集: SSDD_official  928 train / 232 test(val=test)
# 训练配方与 HRSID(712rand)完全一致:300ep / SGD / lr0=0.01 / imgsz512 / batch16 / 从零训练

from ultralytics import YOLO
import os
import torch

MODEL_NAME = "yolo11s-ablation-WEF.yaml"
DATASET_PATH = "../cfg/ssdd_official.yaml"
RUN_NAME = "ssdd_WEF"


def setup_environment():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'


def check_dataset(config_path):
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print("=== 数据集检查 ===")
    print(f"数据集路径: {config['path']}")
    print(f"训练集: {config['train']}  验证/测试集: {config['test']}")
    for split in ('train', 'val'):
        p = os.path.join(config['path'], config[split])
        if os.path.exists(p):
            n = len([f for f in os.listdir(p) if f.lower().endswith(('.bmp', '.jpg', '.png', '.jpeg'))])
            print(f"  {split} 图像数: {n}")
        else:
            print(f"  警告: {split} 路径不存在 - {p}")
    print("==========================\n")


def train_model():
    setup_environment()
    check_dataset(DATASET_PATH)
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"使用 {'GPU: ' + torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'}")
    print(f"模型: {MODEL_NAME}  (run: {RUN_NAME})")
    model = YOLO(MODEL_NAME)
    try:
        results = model.train(
            data=DATASET_PATH,
            epochs=300, imgsz=512, batch=16, workers=0,
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
