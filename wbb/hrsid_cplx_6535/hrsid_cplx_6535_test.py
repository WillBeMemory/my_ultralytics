import torch
from PIL.ImImagePlugin import split

from ultralytics import YOLO
from thop import profile
from torch.utils.flop_counter import FlopCounterMode

# 配置参数
model_path = "./runs/detect/train/weights/best.pt"       # 训练好的模型权重文件
data_yaml = "../cfg/hrsid_cplx_6535.yaml"      # 数据集配置文件（包含验证集路径）
imgsz = 800                              # 图像大小，需与训练一致
# batch = 16                                # 批次大小
# conf = 0.001                              # 置信度阈值（默认0.001）
# iou = 0.6                                  # IoU阈值（默认0.6）
# device = 0                                 # GPU设备（设为'cpu'使用CPU）


if __name__ == '__main__':
    # 加载模型
    model = YOLO(model_path)
    # print(model.model)

    # 运行验证
    results = model.val(
        data=data_yaml,
        split='test', #使用测试集合验证数据A
        imgsz=imgsz,
        # batch=batch,
        # conf=conf,
        # iou=iou,
        # device=device,
        # verbose=False,
        # save=True,
        # save_hybrid=True
    )

    metrics = results.box


    # 辅助函数：确保值是标量
    def to_scalar(x):
        if hasattr(x, 'item'):  # 如果是 NumPy 数组或 PyTorch 张量
            return x.item()
        return x


    # 提取指标并转换为标量
    map50 = to_scalar(metrics.map50)
    map95 = to_scalar(metrics.map)  # map 是 mAP@0.5:0.95
    mp = to_scalar(metrics.mp)  # 平均精确率
    mr = to_scalar(metrics.mr)  # 平均召回率
    f1 = to_scalar(metrics.f1)  # 平均 F1 分数

    # 打印结果
    print("\n========== Validation Results ==========")
    print(f"mAP@0.5:          {map50:.4f}")
    print(f"mAP@0.5:0.95:     {map95:.4f}")
    print(f"Average Precision: {mp:.4f}")
    print(f"Average Recall:    {mr:.4f}")
    print(f"Average F1-score:  {f1:.4f}")
    print("=========================================")

    # 可选：输出每个类别的 AP
    if hasattr(metrics, 'maps') and metrics.maps is not None:
        print("\nPer-class AP@0.5:0.95:")
        for i, ap in enumerate(metrics.maps):
            ap_scalar = to_scalar(ap)
            print(f"Class {i}: {ap_scalar:.4f}")