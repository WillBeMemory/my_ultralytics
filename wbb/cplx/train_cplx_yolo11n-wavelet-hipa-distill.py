from ultralytics.utils.DistillationTrainer import DistillationTrainer

MODEL_NAME = "yolo11n-wavelet-hipa.yaml"
DATASET_PATH = "../cfg/cplx.yaml"
TEACHER_MODEL = "teacher/best.pt"

if __name__ == '__main__':
    # 所有参数（包括蒸馏参数）都在 overrides 字典中
    overrides = {
        "model": MODEL_NAME,
        "data": DATASET_PATH,
        "epochs": 300,
        "imgsz": 800,
        "batch": 8,
        "lr0": 0.001,
        "device": "0",
        "exist_ok": True,
        # 蒸馏参数（在 __init__ 中会被安全 pop 掉）
        "teacher": TEACHER_MODEL,
        "distill_weight": 0.25,
        "temperature": 2.0,
    }

    trainer = DistillationTrainer(overrides=overrides)
    trainer.train()