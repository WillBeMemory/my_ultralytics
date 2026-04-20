from ultralytics import YOLO

# 1. 加载模型（请替换为你的模型文件或YAML配置）
# 如果你已经训练好了模型，直接加载权重文件即可
model = YOLO('./runs/detect/train/weights/best.pt')

# 2. 运行推理，并开启可视化
results = model.predict(r'D:\Study\PostGraduate\YOLO\datasets\HRSID_COMPLEX\images\val\P0021_0_800_7200_8000.jpg',
                        save=True,          # 保存预测结果图
                        visualize=True)     # 开启特征图可视化

print("特征图已生成在 runs/detect/predict/ 目录下")