from ultralytics import YOLO
import torch

model = YOLO("yolo11n.pt") # 或者你训练好的模型
# 创建一个模拟的 640x640 输入 (batch_size=1, channels=3, height=640, width=640)
mock_input = torch.rand(1, 3, 640, 640)

# 遍历 model.model 中的 layer，找到负责处理的特定 Detect 层
# 因为 Detect 层是一个容器，我们需要获取其内部的 model
detect_layer = model.model.model[-1]

# 打印该 Detection 头的步长 (strides)
print(f"模型各输出层的 strides: {detect_layer.stride}")

# 进行前向推理（可选）
results = model(mock_input)