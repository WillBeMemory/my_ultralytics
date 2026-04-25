import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Bottleneck, C3k, Conv
from ultralytics.nn.modules.DepthwiseSeparableConv import WTConv2d

# ===================== 小波滤波器 =====================
def _haar_decompose_filters(in_channels: int):
    dec_lo = torch.tensor([1 / 2, 1 / 2])
    dec_hi = torch.tensor([1 / 2, -1 / 2])
    base = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),  # LL
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),  # LH
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),  # HL
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)   # HH
    ], dim=0)  # (4, 2, 2)
    filters = base[:, None].repeat(1, in_channels, 1, 1)
    return filters.reshape(in_channels * 4, 1, 2, 2)


def _haar_reconstruct_filters(in_channels: int):
    rec_lo = torch.tensor([1., 1.])
    rec_hi = torch.tensor([1., -1.])
    base = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),  # LL
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),  # LH
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),  # HL
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)   # HH
    ], dim=0)  # (4, 2, 2)
    base = base / 2.0  # 归一化
    filters = base[:, None].repeat(1, in_channels, 1, 1)
    return filters.reshape(in_channels * 4, 1, 2, 2)


# ===================== WTConv2d (修正版) =====================
class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dims) * init_scale)

    def forward(self, x):
        return x * self.scale


class WTConv2d(nn.Module):
    """递归小波卷积模块，保持输入输出尺寸与通道数不变"""
    def __init__(self, in_channels, kernel_size=3, depth=2):
        super().__init__()
        self.depth = depth
        self.kernel_size = kernel_size

        dec = _haar_decompose_filters(in_channels)
        rec = _haar_reconstruct_filters(in_channels)
        self.register_buffer('dec_filters', dec)
        self.register_buffer('rec_filters', rec)

        # 基卷积
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=kernel_size // 2, groups=in_channels, bias=False)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        # 每层的子带卷积
        self.wavelet_convs = nn.ModuleList()
        for _ in range(depth):
            self.wavelet_convs.append(nn.ModuleDict({
                'LL': nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size // 2,
                                groups=in_channels, bias=False),
                'LH': nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size // 2,
                                groups=in_channels, bias=False),
                'HL': nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size // 2,
                                groups=in_channels, bias=False),
                'HH': nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size // 2,
                                groups=in_channels, bias=False),
            }))
        self.wavelet_scale = _ScaleModule([1, in_channels, 1, 1], init_scale=0.1)

    def _decompose_recursive(self, x, level):
        if level == self.depth:
            return x, []
        coeffs = F.conv2d(x, self.dec_filters, stride=2, groups=x.shape[1])
        B, C4, H2, W2 = coeffs.shape
        C = C4 // 4
        coeffs = coeffs.view(B, C, 4, H2, W2)
        ll, lh, hl, hh = coeffs[:, :, 0], coeffs[:, :, 1], coeffs[:, :, 2], coeffs[:, :, 3]
        final_ll, high_list = self._decompose_recursive(ll, level + 1)
        high_list.insert(0, {'LH': lh, 'HL': hl, 'HH': hh})
        return final_ll, high_list

    def forward(self, x):
        B, C, H, W = x.shape
        divisor = 2 ** self.depth
        pad_h = (divisor - H % divisor) % divisor
        pad_w = (divisor - W % divisor) % divisor
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate') if pad_h or pad_w else x

        final_ll, high_list = self._decompose_recursive(x_pad, 0)

        # 处理高频
        for level in range(self.depth):
            convs = self.wavelet_convs[level]
            high = high_list[level]
            high_list[level] = {k: convs[k](v) for k, v in high.items()}

        # 处理最深LL
        final_ll = self.wavelet_convs[-1]['LL'](final_ll)

        # 逆重建
        x_wt = final_ll
        for level in reversed(range(self.depth)):
            h = high_list[level]
            combined = torch.stack([x_wt, h['LH'], h['HL'], h['HH']], dim=2)
            B_, C_, _, Hc, Wc = combined.shape
            combined = combined.reshape(B_, C_ * 4, Hc, Wc)
            x_wt = F.conv_transpose2d(combined, self.rec_filters, stride=2, groups=C_)

        x_wt = x_wt[:, :, :H, :W]          # 裁掉padding
        x_wt = self.wavelet_scale(x_wt)

        x_base = self.base_conv(x)
        x_base = self.base_scale(x_base)
        return x_base + x_wt


# ===================== 小波瓶颈 =====================
class WaveletBottleneck(nn.Module):
    """用 WTConv2d 替代标准卷积的瓶颈模块"""
    def __init__(self, c: int, k: int = 3, shortcut: bool = True):
        super().__init__()
        self.wtconv = WTConv2d(c, kernel_size=k, depth=2)  # depth 可根据需求调整
        self.act = nn.SiLU(inplace=True)
        self.add = shortcut

    def forward(self, x):
        out = self.wtconv(x)
        out = self.act(out)
        return x + out if self.add else out


# ===================== 小波C3k =====================
class WaveletC3k(C3k):
    """继承 C3k，但将内部 Bottleneck 替换为 WaveletBottleneck"""
    def __init__(self, c1, c2, n=2, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(WaveletBottleneck(c_, k=k, shortcut=shortcut) for _ in range(n))
        )


# ===================== 最终的 WaveletCSP =====================
class WaveletCSP(nn.Module):
    """
    参数顺序（YOLO 配置中的 args）：
        c2, n, c3k, e, wavelet, k
    其中 c1 由框架自动传入。
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, wavelet=False, k=3):
        super().__init__()
        self.c = int(c2 * e)          # 隐藏通道数
        self.cv1 = Conv(c1, 2 * self.c, 1)

        # 构建子模块
        if wavelet:
            if c3k:
                # 小波增强 + C3k 结构
                self.m = nn.ModuleList(
                    WaveletC3k(self.c, self.c, n=2, shortcut=True, g=1, e=0.5, k=k)
                    for _ in range(n)
                )
            else:
                # 小波增强 + Bottleneck 结构
                self.m = nn.ModuleList(
                    WaveletBottleneck(self.c, k=k, shortcut=True) for _ in range(n)
                )
        else:
            if c3k:
                self.m = nn.ModuleList(
                    C3k(self.c, self.c, n=2, shortcut=True, g=1, e=0.5, k=k) for _ in range(n)
                )
            else:
                self.m = nn.ModuleList(
                    Bottleneck(self.c, self.c, shortcut=True, g=1, k=(k, k), e=1.0) for _ in range(n)
                )

        self.cv2 = Conv((2 + n) * self.c, c2, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # 输入张量 (batch=2, channels=64, spatial=40x40)
    c1 = 64
    x = torch.randn(2, c1, 40, 40).to(device)

    # 测试用例列表
    configs = [
        # (描述, c2, n, c3k, e, wavelet, k)
        ("普通 Bottleneck (无 wavelet, 无 C3k)", 64, 2, False, 0.5, False, 3),
        ("普通 C3k (无 wavelet)", 64, 2, True, 0.5, False, 5),
        ("小波 Bottleneck (wavelet, 无 C3k)", 64, 2, False, 0.5, True, 7),
        ("小波 C3k (wavelet + C3k)", 64, 2, True, 0.5, True, 7),
    ]

    for desc, c2, n, c3k, e, wavelet, k in configs:
        model = WaveletCSP(c1, c2, n=n, c3k=c3k, e=e, wavelet=wavelet, k=k).to(device)
        y = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"--- {desc} ---")
        print(f"    Input : {x.shape}")
        print(f"    Output: {y.shape}  (expected: [{x.shape[0]}, {c2}, 40, 40])")
        print(f"    Params: {params:,}\n")
        # 简单的形状断言
        assert y.shape == (2, c2, 40, 40), f"Shape mismatch for {desc}"

    print("所有测试通过！")