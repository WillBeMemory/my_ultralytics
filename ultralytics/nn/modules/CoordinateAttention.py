import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordAtt(nn.Module):
    """
    坐标注意力模块
    inp: 输入通道数
    oup: 输出通道数（应与输入相同）
    reduction: 压缩率，默认 32
    """
    def __init__(self, inp, oup=None, reduction=32):
        super().__init__()
        if oup is None:
            oup = inp
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        # 水平方向池化 (H, 1)
        x_h = self.pool_h(x)                    # (n, c, h, 1)
        # 垂直方向池化 (1, W)
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # (n, c, w, 1)
        # 拼接并降维
        y = torch.cat([x_h, x_w], dim=2)         # (n, c, h+w, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        # 拆分并恢复维度
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)           # (n, c, 1, w)
        # 生成注意力权重
        a_h = self.conv_h(x_h).sigmoid()         # (n, c, h, 1)
        a_w = self.conv_w(x_w).sigmoid()         # (n, c, 1, w)
        # 广播并加权
        out = identity * a_w * a_h
        return out

if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 创建坐标注意力模块，输入输出通道相同，例如 64
    ca = CoordAtt(inp=64, oup=64).to(device)

    # 生成随机输入 (batch=2, channels=64, height=32, width=32)
    x = torch.randn(2, 64, 32, 32).to(device)

    # 前向传播
    out = ca(x)

    print(f'Input shape : {x.shape}')
    print(f'Output shape: {out.shape}')

    # 简单梯度测试
    loss = out.mean()
    loss.backward()
    print('Backward pass completed.')

    # 检查卷积层是否有梯度
    if ca.conv1.weight.grad is not None:
        print('Gradients exist in conv1')
    else:
        print('No gradients in conv1')