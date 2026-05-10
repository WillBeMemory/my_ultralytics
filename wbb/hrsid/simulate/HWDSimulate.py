import torch

from ultralytics.nn.modules import HWD
from wbb.hrsid.simulate.ModelTester import ModuleTester

# ================== 测试 HWD 模块 ==================
if __name__ == "__main__":
    # 导入 HWD（确保 hwd.py 在同级目录）


    image_path = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\vis\P0001_0_800_7200_8000.jpg"
    label_path = image_path.replace('.jpg', '.txt')

    # 构建待测试的 stem：这里使用 HWD，输入图像是 RGB 3 通道，输出 64 通道
    stem = HWD(c1=3, c2=64)

    tester = ModuleTester(stem=stem,
                          stem_out_channels=64,
                          image_path=image_path,
                          label_path=label_path,
                          input_size=640,
                          device='cuda' if torch.cuda.is_available() else 'cpu')

    tester.train(epochs=300, lr=0.005)
    tester.visualize(conf_thresh=0.5)