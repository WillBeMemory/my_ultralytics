import pickle
import torch
model_path = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\cplx\runs\yolo11n-ternarydpconv\detect\train\weights\best.pt"
import pickle
import torch

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # 对于未知的自定义类，返回一个简单的占位类型
        if name == 'CustomDetectionModel':
            return type('CustomDetectionModel', (), {})
        return super().find_class(module, name)

def load_ckpt_safe(path):
    with open(path, 'rb') as f:
        return SafeUnpickler(f).load()

ckpt = load_ckpt_safe(model_path)
if 'train_args' in ckpt:
    print("训练超参数 train_args:")
    for k, v in ckpt['train_args'].items():
        print(f"  {k}: {v}")
if 'model' in ckpt and hasattr(ckpt['model'], 'args'):
    print("模型配置参数 model.args:")
    for k, v in ckpt['model'].args.items():
        print(f"  {k}: {v}")