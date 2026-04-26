# ============================================
# File: ultralytics/nn/modules/WaveletStem_Denoise.py
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F

__ALL__ = ['WaveletStem']


def haar_filters(in_channels: int):
    """生成 Haar 小波的分解滤波器，形状 (in_channels*4, 1, 2, 2)"""
    dec_lo = torch.tensor([1 / 2, 1 / 2])
    dec_hi = torch.tensor([1 / 2, -1 / 2])
    base = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),  # LL
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),  # LH
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),  # HL
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)  # HH
    ], dim=0)  # (4, 2, 2)
    filters = base[:, None].repeat(1, in_channels, 1, 1)  # (4, C, 2, 2)
    return filters.reshape(in_channels * 4, 1, 2, 2)


def wavelet_decompose(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    """小波分解：输入 (B,C,H,W)，输出 (B,C*4,H/2,W/2)"""
    B, C, H, W = x.shape
    x = F.conv2d(x, filters, stride=2, groups=C, padding=0)
    return x


class SubbandDenoise(nn.Module):
    """
    小波子带差异化去噪模块。
    对四个子带分别采用不同的可学习去噪策略：
    - LL: 可学习衰减系数（软抑制背景，保留轮廓）
    - LH/HL: 共享的标量软阈值（保边去噪）
    - HH: 逐通道软阈值（精准切除噪声尖峰）
    """

    def __init__(self, channels: int, init_ll_alpha: float = 0.8):
        super().__init__()
        self.channels = channels
        # LL 衰减因子初始化为 1.0 (sigmoid后≈0.73，实际更接近1？需调整)
        # 改为初始值 2.0，使 sigmoid(2.0) ≈ 0.88，轻微抑制
        self.ll_alpha = nn.Parameter(torch.tensor(2.0))

        # LH/HL 阈值初始为负值，使 softplus 输出接近 0
        self.lh_hl_thr = nn.Parameter(torch.tensor(-4.0))  # softplus(-4)≈0.018

        # HH 阈值同样初始为负值
        self.hh_thr = nn.Parameter(torch.full((channels,), -4.0))

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coeffs: 小波系数，形状 (B, C,Q 4, H, W)
        Returns:
            去噪后的小波系数，形状相同
        """
        # 拆分四个子带
        ll = coeffs[:, :, 0, :, :]  # (B, C, H, W)
        lh = coeffs[:, :, 1, :, :]
        hl = coeffs[:, :, 2, :, :]
        hh = coeffs[:, :, 3, :, :]

        # LL 子带：可学习衰减（用 sigmoid 限制在 0~1 之间）
        alpha = torch.sigmoid(self.ll_alpha)
        ll = ll * alpha

        # LH 和 HL 子带：共享软阈值（确保阈值非负）
        thr_lh_hl = F.softplus(self.lh_hl_thr)
        lh = torch.sign(lh) * F.relu(torch.abs(lh) - thr_lh_hl)
        hl = torch.sign(hl) * F.relu(torch.abs(hl) - thr_lh_hl)

        # HH 子带：逐通道软阈值
        thr_hh = F.softplus(self.hh_thr).view(1, -1, 1, 1)  # (1, C, 1, 1)
        hh = torch.sign(hh) * F.relu(torch.abs(hh) - thr_hh)

        # 重新堆叠
        return torch.stack([ll, lh, hl, hh], dim=2)


class WaveletStem(nn.Module):
    """
    带子带去噪的小波 Stem 模块，替代 YOLO11 的 P1 和 P2 层。
    输入: (B, 3, H, W)
    输出: (B, out_channels, H/4, W/4)

    内部流程:
        1. 第一次小波分解：3 → 12 通道，尺寸减半
        2. 子带去噪：对 LL/LH/HL/HH 差异化去噪
        3. 中间卷积块：高频特征处理，通道扩张
        4. 第二次小波分解：mid → out_channels，尺寸再减半
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 128, use_denoise: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_denoise = use_denoise

        # 中间通道数需保证第二次小波分解后通道数匹配
        assert out_channels % 4 == 0, "out_channels 必须能被 4 整除"
        mid_channels = out_channels // 4

        # 注册小波滤波器（不可训练）
        self.register_buffer('dec_filters_1', haar_filters(in_channels))
        self.register_buffer('dec_filters_2', haar_filters(mid_channels))

        # 子带去噪模块（作用于第一次分解后）
        if use_denoise:
            self.denoise = SubbandDenoise(channels=in_channels, init_ll_alpha=0.8)
        else:
            self.denoise = None

        # 中间卷积块：标准卷积 + 深度可分离卷积
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels * 4, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 处理奇数尺寸
        _, _, H, W = x.shape
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # 第一次小波分解
        x = wavelet_decompose(x, self.dec_filters_1)  # (B,             12, H/2, W/2)

        # 重塑为 (B, C, 4, H/2, W/2) 以便子带处理
        B, C4, H2, W2 = x.shape
        C = C4 // 4
        x = x.view(B, C, 4, H2, W2)

        # 子带去噪（可选）
        if self.denoise is not None:
            x = self.denoise(x)

        # 展平子带维度回通道维
        x = x.view(B, C * 4, H2, W2)

        # 中间卷积块
        x = self.conv_block(x)  # (B, mid_channels, H/2, W/2)

        # 第二次分解前的尺寸处理
        _, _, H2, W2 = x.shape
        pad_h2 = (2 - H2 % 2) % 2
        pad_w2 = (2 - W2 % 2) % 2
        if pad_h2 > 0 or pad_w2 > 0:
            x = F.pad(x, (0, pad_w2, 0, pad_h2), mode='reflect')

        # 第二次小波分解
        x = wavelet_decompose(x, self.dec_filters_2)  # (B, out_channels, H/4, W/4)

        return x


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WaveletStem(in_channels=3, out_channels=128, use_denoise=True).to(device)
    x = torch.randn(2, 3, 640, 640).to(device)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")  # 期望 (2, 128, 160, 160)

    # 打印去噪参数初始值
    if model.denoise is not None:
        print(f"LL alpha: {torch.sigmoid(model.denoise.ll_alpha).item():.3f}")
        print(f"LH/HL threshold: {F.softplus(model.denoise.lh_hl_thr).item():.3f}")
        print(f"HH thresholds (first 5): {F.softplus(model.denoise.hh_thr[:5]).detach().cpu().numpy()}")

    loss = out.mean()
    loss.backward()
    print("Backward pass completed.")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

