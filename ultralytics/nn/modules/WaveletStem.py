import torch
import torch.nn as nn
import torch.nn.functional as F


# ================== Haar 小波滤波器 ==================
def haar_filters(in_channels: int):
    """生成 Haar 小波分解滤波器 (LL, LH, HL, HH)"""
    dec_lo = torch.tensor([1 / 2, 1 / 2])
    dec_hi = torch.tensor([1 / 2, -1 / 2])
    base = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),  # LL
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),  # LH
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),  # HL
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)   # HH
    ], dim=0)  # (4, 2, 2)
    filters = base[:, None].repeat(1, in_channels, 1, 1)  # (4, C, 2, 2)
    return filters.reshape(in_channels * 4, 1, 2, 2)     # (4C, 1, 2, 2)


def wavelet_decompose(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    """一次 Haar 小波分解，stride=2，输出 (B, 4C, H/2, W/2)"""
    B, C, H, W = x.shape
    return F.conv2d(x, filters, stride=2, groups=C, padding=0)


# ================== 子带增强模块（替换原先的 SubbandDenoise） ==================
class SubbandEnhance(nn.Module):
    """
    对 LL/LH/HL/HH 四个子带进行差异化处理：
    - LL：通道注意力 (SE)
    - LH：水平方向卷积 (3x1) + 软阈值去噪
    - HL：垂直方向卷积 (1x3) + 软阈值去噪
    - HH：1x1 压缩通道 → 1x1 恢复通道（强噪声抑制）
    输出仍保持 (B, C, 4, H, W)
    """
    def __init__(self, channels: int, hh_compress_ratio: int = 2):
        super().__init__()
        # LL: Squeeze-and-Excitation
        self.ll_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(1, channels // 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // 4), channels, 1),
            nn.Sigmoid()
        )

        # LH: 水平方向深度卷积 (kernel=3x1)
        self.lh_conv = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0),
                                 groups=channels, bias=False)
        self.lh_thr = nn.Parameter(torch.full((1, channels, 1, 1), -4.0))

        # HL: 垂直方向深度卷积 (kernel=1x3)
        self.hl_conv = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1),
                                 groups=channels, bias=False)
        self.hl_thr = nn.Parameter(torch.full((1, channels, 1, 1), -4.0))

        # HH: 压缩 + 恢复
        self.hh_compress = nn.Conv2d(channels, channels // hh_compress_ratio, 1, bias=False)
        self.hh_recover = nn.Conv2d(channels // hh_compress_ratio, channels, 1, bias=False)

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        # coeffs: (B, C, 4, H, W)
        ll = coeffs[:, :, 0, :, :]
        lh = coeffs[:, :, 1, :, :]
        hl = coeffs[:, :, 2, :, :]
        hh = coeffs[:, :, 3, :, :]

        # LL 增强
        ll = ll * self.ll_se(ll)

        # LH 方向增强 + 软阈值
        lh = self.lh_conv(lh)
        thr_lh = F.softplus(self.lh_thr)
        lh = torch.sign(lh) * F.relu(torch.abs(lh) - thr_lh)

        # HL 方向增强 + 软阈值
        hl = self.hl_conv(hl)
        thr_hl = F.softplus(self.hl_thr)
        hl = torch.sign(hl) * F.relu(torch.abs(hl) - thr_hl)

        # HH 强压缩去噪
        hh = self.hh_compress(hh)   # (B, C//r, H, W)
        hh = self.hh_recover(hh)    # (B, C, H, W)

        return torch.stack([ll, lh, hl, hh], dim=2)  # (B, C, 4, H, W)


# ================== 修改后的 WaveletStem ==================
class WaveletStem(nn.Module):
    """
    小波分解 + 差异化子带增强 + 轻量卷积投影。
    输入: (B, in_channels, H, W)
    输出: (B, out_channels, H/2, W/2)
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 128,
                 use_denoise: bool = True, hh_compress_ratio: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_denoise = use_denoise

        # 注册 Haar 分解滤波器
        self.register_buffer('dec_filters', haar_filters(in_channels))

        # 子带增强模块（替换原先的 SubbandDenoise）
        if use_denoise:
            self.denoise = SubbandEnhance(channels=in_channels,
                                          hh_compress_ratio=hh_compress_ratio)
        else:
            self.denoise = None

        # 投影卷积：将 4*C 通道映射到 out_channels
        # 深度可分离卷积 + 1x1 压缩，轻量且保留通道间交互
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels * 4, 3, padding=1,
                      groups=in_channels * 4, bias=False),
            nn.Conv2d(in_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 处理奇数尺寸：填充为偶数
        _, _, H, W = x.shape
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # 一次小波分解
        coeffs = wavelet_decompose(x, self.dec_filters)  # (B, C*4, H/2, W/2)
        B, C4, H2, W2 = coeffs.shape
        C = C4 // 4
        coeffs = coeffs.view(B, C, 4, H2, W2)

        # 子带差异化增强
        if self.denoise is not None:
            coeffs = self.denoise(coeffs)  # (B, C, 4, H2, W2)

        # 展平为 (B, 4C, H2, W2) 并通过投影卷积
        x = coeffs.reshape(B, C * 4, H2, W2)
        return self.conv_block(x)


# ================== 简单 main 测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 构造随机输入 (模拟 RGB 图像 640x640)
    x = torch.randn(2, 3, 640, 640).to(device)

    # 创建模型
    model = WaveletStem(in_channels=3, out_channels=128, use_denoise=True).to(device)
    model.eval()

    with torch.no_grad():
        out = model(x)

    print(f"Input shape : {x.shape}")
    print(f"Output shape: {out.shape}")   # 期望 (2, 128, 320, 320)
    expected_shape = (2, 128, 320, 320)
    assert out.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {out.shape}"
    print("✅ Output shape verified.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")