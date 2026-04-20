import torch
import torch.nn as nn
import torch.nn.functional as F


class MDCA(nn.Module):
    """
    多方向坐标注意力 (Multi-Directional Coordinate Attention)
    在原始CA基础上增加对角线方向池化，增强方向感知能力
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # 四个方向共享的1x1卷积层，用于降维和信息融合
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        # 四个方向分别使用独立的1x1卷积生成注意力权重
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1)
        self.conv_d1 = nn.Conv2d(mip, in_channels, kernel_size=1)  # 45° 方向
        self.conv_d2 = nn.Conv2d(mip, in_channels, kernel_size=1)  # 135° 方向

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x

        # 1. 四个方向的特征编码
        # 水平 (H,1)
        x_h = torch.mean(x, dim=3, keepdim=True)
        # 垂直 (1,W)
        x_w = torch.mean(x, dim=2, keepdim=True).permute(0, 1, 3, 2)

        # 对角线方向: 使用旋转特征图的方法
        # 45°方向: 旋转45度后做垂直池化
        x_d1 = torch.rot90(x, k=-1, dims=[2, 3])  # 逆时针旋转90°
        x_d1 = torch.mean(x_d1, dim=2, keepdim=True)
        x_d1 = torch.rot90(x_d1, k=1, dims=[2, 3])  # 转回来

        # 135°方向: 旋转90度后做垂直池化
        x_d2 = torch.rot90(x, k=1, dims=[2, 3])  # 顺时针旋转90°
        x_d2 = torch.mean(x_d2, dim=2, keepdim=True)
        x_d2 = torch.rot90(x_d2, k=-1, dims=[2, 3])  # 转回来

        # 2. 拼接四个方向的特征
        y = torch.cat([x_h, x_w, x_d1, x_d2], dim=2)  # (B, C, H+W+H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 3. 拆分为四个方向，并恢复原始通道
        split_sizes = [H, W, H, W]
        x_h, x_w, x_d1, x_d2 = torch.split(y, split_sizes, dim=2)

        x_w = x_w.permute(0, 1, 3, 2)
        x_d1 = x_d1.permute(0, 1, 3, 2)
        x_d2 = x_d2.permute(0, 1, 3, 2)

        # 4. 生成四个方向的注意力权重
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        a_d1 = torch.sigmoid(self.conv_d1(x_d1))
        a_d2 = torch.sigmoid(self.conv_d2(x_d2))

        # 5. 融合四个方向的注意力
        out = identity * (a_w * a_h * a_d1 * a_d2)
        return out

