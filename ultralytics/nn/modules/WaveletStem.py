import torch
import torch.nn as nn
import torch.nn.functional as F


def haar_filters(in_channels: int):
    """
    生成形状为 (in_channels*4, 1, 2, 2) 的 Haar 小波滤波器，
    保证 groups=in_channels 时每个通道独立分解为 [LL, LH, HL, HH]。
    """
    dec_lo = torch.tensor([1 / 2, 1 / 2])
    dec_hi = torch.tensor([1 / 2, -1 / 2])
    base_LL = (dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    base_LH = (dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    base_HL = (dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    base_HH = (dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    per_channel = torch.cat([base_LL, base_LH, base_HL, base_HH], dim=1)  # (1,4,2,2)
    per_channel = per_channel.repeat(in_channels, 1, 1, 1)
    return per_channel.reshape(in_channels * 4, 1, 2, 2)


def wavelet_decompose(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    """用给定滤波器对 x 做 stride=2 的深度卷积，实现 Haar 小波分解"""
    B, C, H, W = x.shape
    return F.conv2d(x, filters, stride=2, groups=C, padding=0)


class WaveletStem(nn.Module):
    """
    纯小波下采样茎（无标准分支，所有子带使用标准卷积且有通道交互）：
    - Haar 小波分解为 LL, LH, HL, HH
    - LL : 标准 3×3 卷积 + BN + SiLU
    - LH : 垂直条状卷积 (1,3) + BN + SiLU  （增强水平边缘）
    - HL : 水平条状卷积 (3,1) + BN + SiLU  （增强垂直边缘）
    - HH : 对角初始化 3×3 卷积 + BN + SiLU
    - 拼接后 1×1 投影到 out_channels
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 固定 Haar 分解滤波器
        self.register_buffer('dec_filters', haar_filters(in_channels))

        # LL：标准 3x3 卷积
        self.ll_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True)
        )

        # LH：垂直条状卷积 (1,3)，padding 保持空间尺寸
        self.lh_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True)
        )

        # HL：水平条状卷积 (3,1)
        self.hl_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True)
        )

        # HH：3x3 卷积，对角初始化
        self.hh_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True)
        )
        self._init_hh_diag()

        # 1x1 投影层，将 4*in_channels 压缩到 out_channels
        self.proj = nn.Conv2d(in_channels * 4, out_channels, 1, bias=False)
        self.proj_bn = nn.BatchNorm2d(out_channels)
        self.proj_act = nn.SiLU(inplace=True)

    def _init_hh_diag(self):
        """初始化 HH 卷积核为主对角线正权、反对角线负权"""
        with torch.no_grad():
            weight = self.hh_conv[0].weight  # (in_channels, in_channels, 3, 3)
            nn.init.zeros_(weight)
            # 对每个输出通道 i，赋予对角结构
            for i in range(self.in_channels):
                # 主对角线
                weight[i, i, 0, 0] = 1.0
                weight[i, i, 1, 1] = 2.0
                weight[i, i, 2, 2] = 1.0
                # 反对角线
                weight[i, i, 0, 2] = -1.0
                weight[i, i, 2, 0] = -1.0
            # 按输出通道归一化
            weight /= (weight.view(self.in_channels, -1).norm(p=2, dim=1).view(self.in_channels, 1, 1, 1) + 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # 填充至偶数尺寸（Haar 分解要求）
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # 小波分解
        coeffs = wavelet_decompose(x, self.dec_filters)          # (B, 4*C, H/2, W/2)
        _, C4, H2, W2 = coeffs.shape
        coeffs = coeffs.view(B, self.in_channels, 4, H2, W2)

        ll = coeffs[:, :, 0]
        lh = coeffs[:, :, 1]
        hl = coeffs[:, :, 2]
        hh = coeffs[:, :, 3]

        # 各子带通过对应的标准卷积（有通道交互）
        ll = self.ll_conv(ll)   # (B, in_ch, H/2, W/2)
        lh = self.lh_conv(lh)
        hl = self.hl_conv(hl)
        hh = self.hh_conv(hh)

        # 拼接
        combined = torch.cat([ll, lh, hl, hh], dim=1)   # (B, 4*in_ch, H/2, W/2)
        out = self.proj(combined)
        return self.proj_act(self.proj_bn(out))


# ================== 测试 ==================
# 修改后的测试部分
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    x_even = torch.randn(2, 3, 640, 640).to(device)
    x_odd  = torch.randn(2, 3, 641, 641).to(device)

    model = WaveletStem(in_channels=3, out_channels=64).to(device)
    model.eval()

    with torch.no_grad():
        out_even = model(x_even)
        out_odd  = model(x_odd)

    print(f"Even input: {x_even.shape} -> output: {out_even.shape}")
    print(f"Odd input: {x_odd.shape} -> output: {out_odd.shape}")

    # 修正断言：奇数输入填充后偶数化，尺寸减半
    assert out_even.shape == (2, 64, 320, 320), f"Shape mismatch: {out_even.shape}"
    assert out_odd.shape  == (2, 64, 321, 321), f"Shape mismatch: {out_odd.shape}"
    print("✅ Output shapes verified.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")