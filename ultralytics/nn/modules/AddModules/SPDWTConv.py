import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def create_haar_filters(channels, dtype=torch.float):
    """生成 Haar 小波分解和重构滤波器 (4*C, 1, 2, 2)"""
    lo = torch.tensor([1.0, 1.0]) / math.sqrt(2)
    hi = torch.tensor([1.0, -1.0]) / math.sqrt(2)
    # 四个二维滤波器 (2x2)
    LL = lo.unsqueeze(0) * lo.unsqueeze(1)
    LH = lo.unsqueeze(0) * hi.unsqueeze(1)
    HL = hi.unsqueeze(0) * lo.unsqueeze(1)
    HH = hi.unsqueeze(0) * hi.unsqueeze(1)
    # 堆叠为 (4, 2, 2) -> (4, 1, 2, 2) -> (4*C, 1, 2, 2)
    dec_filters = torch.stack([LL, LH, HL, HH])      # (4, 2, 2)
    dec_filters = dec_filters.unsqueeze(1)            # (4, 1, 2, 2)
    dec_filters = dec_filters.repeat(channels, 1, 1, 1)  # (4*C, 1, 2, 2)
    rec_filters = dec_filters.clone()                 # 重构滤波器相同
    return dec_filters, rec_filters


class HaarWTConv2d(nn.Module):
    """纯 PyTorch 实现的多级 Haar 小波卷积（无任何外部依赖）"""
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, wt_levels=2, bias=True):
        super().__init__()
        assert in_channels == out_channels, "WTConv2d 要求 in_channels == out_channels"
        self.channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride

        # 固定 Haar 滤波器
        dec_filters, rec_filters = create_haar_filters(in_channels)
        self.register_buffer('dec_filters', dec_filters)  # (4*C, 1, 2, 2)
        self.register_buffer('rec_filters', rec_filters)

        # 基础卷积分支（原分辨率深度可分离卷积）
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same',
                                   groups=in_channels, bias=bias)
        self.base_scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))

        # 各级子带卷积（分组数 = 4*C，每个子带独立进行空间卷积）
        self.wavelet_convs = nn.ModuleList()
        self.wavelet_scales = nn.ParameterList()
        for _ in range(wt_levels):
            conv = nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same',
                             groups=in_channels * 4, bias=False)
            self.wavelet_convs.append(conv)
            # 初始化尺度为 0.1，让网络初期更多依赖基础支路
            self.wavelet_scales.append(nn.Parameter(torch.ones(1, in_channels * 4, 1, 1) * 0.1))

        # stride > 1 时使用平均池化（而非步长卷积）
        if stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def _dwt(self, x):
        B, C, H, W = x.shape
        # 确保尺寸为偶数（Haar 分解要求）
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        # 分组卷积实现小波分解（groups=C）
        coeffs = F.conv2d(x, self.dec_filters, stride=2, groups=C)
        # 输出形状 (B, 4*C, H/2, W/2) → 重组为 (B, C, 4, H/2, W/2)
        return coeffs.view(B, C, 4, coeffs.shape[-2], coeffs.shape[-1])

    def _idwt(self, coeffs, target_h, target_w):
        B, C, _, h, w = coeffs.shape
        flat = coeffs.reshape(B, C * 4, h, w)
        # 转置卷积实现逆变换（groups=C）
        out = F.conv_transpose2d(flat, self.rec_filters, stride=2, groups=C)
        return out[:, :, :target_h, :target_w]

    def forward(self, x):
        B, C, H, W = x.shape
        orig_h, orig_w = H, W

        # -------- 多级分解与子带卷积 --------
        levels = []
        current = x
        for lvl in range(self.wt_levels):
            coeffs = self._dwt(current)                    # (B, C, 4, h, w)
            # 所有子带展平后卷积
            flat = coeffs.reshape(B, C * 4, coeffs.shape[-2], coeffs.shape[-1])
            conv_out = self.wavelet_convs[lvl](flat) * self.wavelet_scales[lvl]
            processed = conv_out.view(B, C, 4, coeffs.shape[-2], coeffs.shape[-1])
            levels.append(processed)
            current = processed[:, :, 0]                   # LL 作为下一级输入

        # -------- 逐级重构 --------
        for lvl in reversed(range(self.wt_levels)):
            if lvl == self.wt_levels - 1:
                ll = current
            else:
                ll = current
            lh = levels[lvl][:, :, 1]
            hl = levels[lvl][:, :, 2]
            hh = levels[lvl][:, :, 3]
            coeffs = torch.stack([ll, lh, hl, hh], dim=2)  # (B, C, 4, h, w)
            target_h = levels[lvl].shape[-2] * 2
            target_w = levels[lvl].shape[-1] * 2
            if lvl == 0:
                target_h, target_w = orig_h, orig_w
            current = self._idwt(coeffs, target_h, target_w)

        # -------- 基础卷积支路（保留原始细节）--------
        base = self.base_conv(x) * self.base_scale
        if self.base_conv.bias is not None:
            base = base + self.base_conv.bias.view(1, -1, 1, 1)

        out = base + current
        if self.do_stride is not None:
            out = self.do_stride(out)
        return out


class SPDWTConv(nn.Module):
    """
    SPD + WTConv 下采样模块
    - pixel_unshuffle 无损空间到深度
    - 纯 PyTorch 多级 Haar 小波卷积（所有子带处理）
    - 1×1 卷积压缩到输出通道

    Args:
        c1: 输入通道
        c2: 输出通道
        k: 压缩卷积核大小 (默认 1)
        s: 步长 (2 为下采样模式，1 退化为普通卷积)
        kernel_size: 小波卷积内部核大小 (默认 5)
        wt_levels: 小波分解级数 (推荐 1~3)
    """
    def __init__(self, c1, c2, k=1, s=2, kernel_size=5, wt_levels=2):
        super().__init__()
        self.s = s
        if s == 2:
            c_mid = c1 * 4  # SPD 膨胀后的通道数
            self.wtconv = HaarWTConv2d(in_channels=c_mid, out_channels=c_mid,
                                       kernel_size=kernel_size, stride=1,
                                       wt_levels=wt_levels, bias=True)
            # 1×1 压缩到目标通道
            self.compress = nn.Conv2d(c_mid, c2, kernel_size=k, stride=1,
                                     padding=autopad(k), bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = nn.SiLU(inplace=True)
        else:
            # 普通卷积（无 SPD）
            self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s,
                                 padding=autopad(k), bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        if self.s == 2:
            B, C, H, W = x.shape
            # 确保空间尺寸为偶数
            pad_h = (2 - H % 2) % 2
            pad_w = (2 - W % 2) % 2
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            # SPD 无损下采样
            x = F.pixel_unshuffle(x, 2)          # (B, 4C, H/2, W/2)
            # 多级小波卷积（不改变尺寸）
            x = self.wtconv(x)
            # 通道压缩
            x = self.compress(x)
            return self.act(self.bn(x))
        else:
            return self.act(self.bn(self.conv(x)))


# ---------- 简单测试 ----------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = SPDWTConv(c1=64, c2=128, k=3, s=2, kernel_size=5, wt_levels=2).to(device)
    model.train()
    x = torch.randn(2, 64, 32, 32).to(device)
    y = model(x)
    print(f"Input: {x.shape} → Output: {y.shape} (expected [2,128,16,16])")
    loss = y.mean()
    loss.backward()
    print("Gradients OK")