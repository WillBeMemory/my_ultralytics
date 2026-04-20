
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

_all_ = ['HWD']

class HWD(nn.Module):
    def __init__(self, c1, c2):
        # print(f"HWD __init__ called with: c1={c1}, c2={c2}, args={args}, kwargs={kwargs}")
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        wavelet_kernels = torch.tensor([
            [[[1, 1], [1, 1]]],
            [[[1, 1], [-1, -1]]],
            [[[1, -1], [1, -1]]],
            [[[1, -1], [-1, 1]]]
        ], dtype=torch.float32) / 2.0
        self.register_buffer('weight', wavelet_kernels.repeat(c1, 1, 1, 1))
        self.conv = nn.Conv2d(4 * c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        # 可选断言
        # assert C == self.c1, f"Input channels {C} != expected {self.c1}"
        x_wavelet = F.conv2d(x, self.weight, stride=2, groups=self.c1)
        out = self.conv(x_wavelet)
        out = self.bn(out)
        out = self.act(out)
        return out

# class HWD(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(HWD, self).__init__()
#         self.wt = DWTForward(J=1, mode='zero', wave='haar')
#         self.conv_bn_relu = nn.Sequential(
#             nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         yL, yH = self.wt(x)
#         y_HL = yH[0][:, :, 0, ::]
#         y_LH = yH[0][:, :, 1, ::]
#         y_HH = yH[0][:, :, 2, ::]
#         x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
#         x = self.conv_bn_relu(x)
#         return x

# 输入 B C H W,  输出 B C H W
if __name__ == '__main__':
    block = HWD(64, 64)  # 输入通道数，输出通道数
    input = torch.rand(3, 64, 64, 64)
    output = block(input)
    print(output.size())