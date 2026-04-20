import torch
from ultralytics import YOLO
import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv, C3k2, SPPF, C2PSA
from ultralytics.nn.modules.DepthwiseSeparableConv import DepthwiseSeparableConvWithWTConv2d  # 替换为实际导入

if __name__ == '__main__':

    # model = YOLO('yolo11n-dypconv.yaml')
    # x = torch.randn(1, 3, 640, 640)
    # with torch.no_grad():
    #     # 假设你的模块在 backbone 的第 7 个层（索引7）
    #     print(model.model.model[7].output_shape)  # 需要模块记录输出形状，或直接前向

    # 模拟 backbone
    x = torch.randn(1, 3, 640, 640)

    # 层0
    conv0 = Conv(3, 64, 3, 2)
    x = conv0(x)
    print("0:", x.shape)  # (1,64,320,320)

    # 层1
    conv1 = Conv(64, 128, 3, 2)
    x = conv1(x)
    print("1:", x.shape)  # (1,128,160,160)

    # 层2
    c3k2_2 = C3k2(128, 256, 2, False, 0.25)
    x = c3k2_2(x)
    print("2:", x.shape)  # (1,256,160,160)

    # 层3
    conv3 = Conv(256, 256, 3, 2)
    x = conv3(x)
    print("3:", x.shape)  # (1,256,80,80)

    # 层4
    c3k2_4 = C3k2(256, 512, 2, False, 0.25)
    x = c3k2_4(x)
    print("4:", x.shape)  # (1,512,80,80)

    # 层5
    conv5 = Conv(512, 512, 3, 2)
    x = conv5(x)
    print("5:", x.shape)  # (1,512,40,40)

    # 层6
    c3k2_6 = C3k2(512, 512, 2, True)
    x = c3k2_6(x)
    print("6:", x.shape)  # (1,512,40,40)

    # 层7 (你的模块)
    custom = DepthwiseSeparableConvWithWTConv2d(512, 1024, k=3, s=2)
    x = custom(x)
    print("7:", x.shape)  # 应为 (1,1024,20,20)

    # 层8
    c3k2_8 = C3k2(1024, 1024, 2, True)
    x = c3k2_8(x)
    print("8:", x.shape)  # (1,1024,20,20)

    # 层9
    sppf = SPPF(1024, 1024, 5)
    x = sppf(x)
    print("9:", x.shape)  # (1,1024,20,20)

    # 层10
    c2psa = C2PSA(1024, 1024)
    x = c2psa(x)
    print("10:", x.shape)  # (1,1024,20,20)