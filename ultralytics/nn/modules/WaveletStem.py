import torch
import torch.nn as nn
import torch.nn.functional as F


# ================== Haar 小波滤波器 ==================
def haar_filters(in_channels: int):
    dec_lo = torch.tensor([1 / 2, 1 / 2])
    dec_hi = torch.tensor([1 / 2, -1 / 2])
    base = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),  # LL
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),  # LH
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),  # HL
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)   # HH
    ], dim=0)  # (4, 2, 2)
    filters = base[:, None].repeat(1, in_channels, 1, 1)  # (4, C, 2, 2)
    return filters.reshape(in_channels * 4, 1, 2, 2)


def wavelet_decompose(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    return F.conv2d(x, filters, stride=2, groups=C, padding=0)


# ================== 方向性子带卷积 ==================
class DirectionalConv(nn.Module):
    """
    根据不同子带特性设计的方向卷积：
    - LL: 标准 3×3 深度卷积
    - LH: (1, 3) 垂直条状深度卷积（增强水平边缘）
    - HL: (3, 1) 水平条状深度卷积（增强垂直边缘）
    - HH: 3×3 深度卷积，可初始化为对角模式（强化对角边缘）
    """
    def __init__(self, channels: int, mode: str, use_diag_init: bool = True):
        super().__init__()
        self.mode = mode
        if mode == 'll':
            self.conv = nn.Conv2d(channels, channels, 3, padding=1,
                                  groups=channels, bias=False)
        elif mode == 'lh':
            self.conv = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1),
                                  groups=channels, bias=False)
        elif mode == 'hl':
            self.conv = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0),
                                  groups=channels, bias=False)
        elif mode == 'hh':
            self.conv = nn.Conv2d(channels, channels, 3, padding=1,
                                  groups=channels, bias=False)
            if use_diag_init:
                self._init_diag_weights()
        else:
            raise ValueError(f"Invalid mode: {mode}")
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def _init_diag_weights(self):
        """初始化 HH 卷积核为对角增强模式（主对角线正，反对角线负）"""
        with torch.no_grad():
            weight = self.conv.weight  # (C, 1, 3, 3)
            nn.init.zeros_(weight)
            # 主对角线 (+)
            weight[:, 0, 0, 0] = 1.0
            weight[:, 0, 1, 1] = 1.0
            weight[:, 0, 2, 2] = 1.0
            # 反对角线 (-)
            weight[:, 0, 0, 2] = -0.5
            weight[:, 0, 1, 1] -= 1.0   # 中心点被正对角线占用，适当调整
            weight[:, 0, 2, 0] = -0.5
            # 归一化
            C = weight.shape[0]
            weight /= weight.view(C, -1).norm(p=2, dim=1).view(C, 1, 1, 1) + 1e-6

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ================== 完整的 WaveletStem（无阈值） ==================
class WaveletStem(nn.Module):
    """
    小波分解 + 四种方向卷积 + 通道扩张。
    输入: (B, in_channels, H, W)
    输出: (B, out_channels, H/2, W/2)
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 64,
                 use_diag_init: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 注册 Haar 分解滤波器
        self.register_buffer('dec_filters', haar_filters(in_channels))

        # 四个子带专用的方向卷积
        self.ll_conv = DirectionalConv(in_channels, 'll')
        self.lh_conv = DirectionalConv(in_channels, 'lh')
        self.hl_conv = DirectionalConv(in_channels, 'hl')
        self.hh_conv = DirectionalConv(in_channels, 'hh', use_diag_init=use_diag_init)

        # 通道扩张：4*C → out_channels
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        # 填充到偶数尺寸
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # 小波分解
        coeffs = wavelet_decompose(x, self.dec_filters)  # (B, 4C, H/2, W/2)
        B, C4, H2, W2 = coeffs.shape
        C = C4 // 4
        coeffs = coeffs.view(B, C, 4, H2, W2)

        # 分离子带
        ll = coeffs[:, :, 0, :, :]
        lh = coeffs[:, :, 1, :, :]
        hl = coeffs[:, :, 2, :, :]
        hh = coeffs[:, :, 3, :, :]

        # 分别进行方向卷积增强
        ll = self.ll_conv(ll)
        lh = self.lh_conv(lh)
        hl = self.hl_conv(hl)
        hh = self.hh_conv(hh)

        # 拼接并扩张通道
        out = torch.cat([ll, lh, hl, hh], dim=1)   # (B, 4C, H/2, W/2)
        out = self.expand(out)                     # (B, out_channels, H/2, W/2)
        return out


# ================== 简单 main 测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 模拟输入 RGB 640x640
    x = torch.randn(2, 3, 640, 640).to(device)

    # 构建模型（输出 64 通道，也可测试 128）
    model = WaveletStem(in_channels=3, out_channels=64, use_diag_init=True).to(device)
    model.eval()

    with torch.no_grad():
        out = model(x)

    print(f"Input shape : {x.shape}")
    print(f"Output shape: {out.shape}")   # 期望 (2, 64, 320, 320)
    expected_shape = (2, 64, 320, 320)
    assert out.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {out.shape}"
    print("✅ Output shape verified.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")