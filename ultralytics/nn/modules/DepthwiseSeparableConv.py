import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DepthwiseSeparableConvWithWTConv2d']

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dims) * init_scale)
    def forward(self, x):
        return x * self.scale

def haar_filters(in_channels):
    """
    生成 Haar 小波的分解和重构滤波器，两者形状相同： (in_channels*4, 1, 2, 2)
    """
    # 基础滤波器 (2x2)
    dec_lo = torch.tensor([1/2, 1/2])
    dec_hi = torch.tensor([1/2, -1/2])
    # 四个方向滤波器 (4, 2, 2)
    base = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
    ], dim=0)  # (4,2,2)
    # 扩展通道维度
    filters = base[:, None].repeat(1, in_channels, 1, 1)  # (4, C, 2, 2)
    filters = filters.reshape(in_channels * 4, 1, 2, 2)   # (4*C, 1, 2, 2)
    # 分解和重构使用相同滤波器
    return filters, filters  # dec_filters, rec_filters

@torch.jit.script
def wavelet_transform(x, filters):
    """输入 (B,C,H,W)，输出 (B,C,4,H/2,W/2)"""
    B, C, H, W = x.shape
    # 使用普通卷积，groups=C，滤波器 (4*C,1,2,2)
    x = F.conv2d(x, filters, stride=2, groups=C, padding=0)  # (B,4*C,H/2,W/2)
    x = x.view(B, C, 4, x.shape[-2], x.shape[-1])
    return x

@torch.jit.script
def inverse_wavelet_transform(x, filters):
    """输入 (B,C,4,H/2,W/2)，输出 (B,C,H,W)"""
    B, C, _, Hh, Wh = x.shape
    x = x.view(B, C * 4, Hh, Wh)  # (B,4*C,H/2,W/2)
    # 使用转置卷积，groups=C，滤波器 (4*C,1,2,2)
    x = F.conv_transpose2d(x, filters, stride=2, groups=C, padding=0)  # (B,C,H,W)
    return x

class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        assert in_channels == out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        dec_filters, rec_filters = haar_filters(in_channels)
        self.register_buffer('dec_filters', dec_filters)   # (4*C,1,2,2)
        self.register_buffer('rec_filters', rec_filters)   # (4*C,1,2,2)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=kernel_size//2, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_conv = nn.Conv2d(in_channels * 4, in_channels * 4,
                                      kernel_size, padding=kernel_size//2,
                                      groups=in_channels * 4, bias=False)
        self.wavelet_scale = _ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1)

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = (2 - H % 2) % 2  # 若 H 为奇数则 pad_h=1，否则 0
        pad_w = (2 - W % 2) % 2  # 若 W 为奇数则 pad_w=1，否则 0
        if pad_h or pad_w:
            # 使用 reflect 模式填充至偶数尺寸
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]
        else:
            x_pad = x
            H_pad, W_pad = H, W

        # 小波分支
        x_wt = wavelet_transform(x_pad, self.dec_filters)  # (B,C,4,H/2,W/2)
        x_wt = x_wt.reshape(B, C * 4, H_pad // 2, W_pad // 2)  # (B,4C,H/2,W/2)
        x_wt = self.wavelet_conv(x_wt)
        x_wt = self.wavelet_scale(x_wt)
        x_wt = x_wt.reshape(B, C, 4, H_pad // 2, W_pad // 2)  # (B,C,4,H/2,W/2)
        x_wt = inverse_wavelet_transform(x_wt, self.rec_filters)  # (B,C,H_pad,W_pad)
        if pad_h or pad_w:
            x_wt = x_wt[:, :, :H, :W]  # 裁剪回原始尺寸

        # 普通分支
        x_base = self.base_conv(x)
        x_base = self.base_scale(x_base)

        return x_base + x_wt

class DepthwiseSeparableConvWithWTConv2d(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.depthwise = WTConv2d(c1, c1, kernel_size=k, bias=False)
        self.pointwise = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.downsample = None
        if s > 1:
            self.downsample = nn.AvgPool2d(s, stride=s)

    def forward(self, x):
        x = self.depthwise(x)
        if self.downsample:
            x = self.downsample(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x

# 测试
if __name__ == "__main__":

    # 测试 stride=2 的模块
    model = DepthwiseSeparableConvWithWTConv2d(64, 128, k=3, s=2)
    x = torch.randn(1, 64, 80, 80)  # 模拟 P4 输出尺寸 80x80（输入 640，P4 为 40x40? 这里用 80 演示）
    out = model(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", out.shape)  # 应输出 (1,128,40,40) 即宽高减半