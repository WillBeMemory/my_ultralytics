import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarWaveletUpsample(nn.Module):
    """
    小波特征上采样模块 (Haar Wavelet Upsample)

    通过可学习的卷积从低分辨率特征生成高频子带 (LH, HL, HH)，
    与原始高分辨率特征 (作为 LL) 结合，利用逆 Haar 小波变换重建 2 倍上采样特征图。

    输入:
        low  : (B, C_low, H, W)     当前层的高分辨率特征 (如 P4 层的特征)
        high : (B, C_high, H/2, W/2) 上一层的低分辨率特征 (如 P5 层上采样前的特征)

    输出:
        (B, C_low, 2*H, 2*W)       上采样后的特征图，通道数与 low 的通道数一致

    参数:
        low_ch  (int): low 特征的通道数 (C_low)
        high_ch (int): high 特征的通道数 (C_high)
    """
    def __init__(self, low_ch: int, high_ch: int):
        super().__init__()
        self.low_ch = low_ch
        self.high_ch = high_ch

        # 将 high 特征上采样 2 倍，使其尺寸与 low 对齐
        self.high_up = nn.Upsample(scale_factor=2, mode='nearest')

        # 融合 low 和上采样后的 high，生成三个高频子带 (LH, HL, HH)
        # 输出通道 3*low_ch，对应三个子带（每个子带通道数为 low_ch）
        self.fuse = nn.Sequential(
            nn.Conv2d(low_ch + high_ch, low_ch * 3, 3, padding=1, bias=False),
            nn.BatchNorm2d(low_ch * 3),
            nn.SiLU(inplace=True)
        )

        # 最终输出微调卷积
        self.out_conv = nn.Conv2d(low_ch, low_ch, 1, bias=False)

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        # 1. 将高层特征上采样到 low 的尺寸
        high_up = self.high_up(high)                     # (B, C_high, H, W)

        # 2. 拼接 low 与 high_up，并生成三个高频子带特征
        combined = torch.cat([low, high_up], dim=1)      # (B, C_low + C_high, H, W)
        subbands = self.fuse(combined)                   # (B, 3*C_low, H, W)

        # 3. 拆分为三个子带: LH, HL, HH，每个形状 (B, C_low, H, W)
        lh, hl, hh = subbands.chunk(3, dim=1)

        # 4. 执行逆 Haar 小波变换 (IDWT)
        # 将 LL(low), LH, HL, HH 堆叠为 (B, C_low, 4, H, W)
        stacked = torch.stack([low, lh, hl, hh], dim=2)  # (B, C_low, 4, H, W)
        B, C, _, H, W = stacked.shape

        # 重排为 (B*C_low, 4, H, W) 以使用 pixel_shuffle 实现 IDWT
        combined = stacked.reshape(B * C, 4, H, W)
        # Pixel Shuffle: (B*C, 4, H, W) -> (B*C, 1, 2H, 2W) -> (B*C, 4, H*2, W*2) 实际上 pixel_shuffle 将通道维的空间信息重排
        # 对于 Haar 逆变换，4 个子带应当形成 2x2 块，pixel_shuffle 恰好可以完美重组
        up = F.pixel_shuffle(combined, 2)                # (B*C, 1, 2*H, 2*W)

        # 恢复 batch 和通道维
        out = up.reshape(B, C, 2 * H, 2 * W)             # (B, C_low, 2H, 2W)

        # 最终调整
        return self.out_conv(out)


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 模拟 P4 (low) 和 P5 (high) 的特征图
    # P4: 128 通道, 40x40
    # P5: 256 通道, 20x20 (上采样后应得到 40x40，最终输出 80x80？不对，这里 low 是 P4，上采样后应得到 P3 尺寸？)
    # 我们简单测试：从 20x20 上采样到 40x40
    low  = torch.randn(2, 128, 40, 40).to(device)   # 模拟 P4
    high = torch.randn(2, 256, 20, 20).to(device)   # 模拟 P5

    model = HaarWaveletUpsample(low_ch=128, high_ch=256).to(device)
    out = model(low, high)

    print(f"low  shape: {low.shape}")
    print(f"high shape: {high.shape}")
    print(f"out  shape: {out.shape}")   # 期望: (2, 128, 80, 80) ？不对，这里是 low (40x40) 作为 LL，上采样后应得到 80x80？实际上 low 本身就是高分辨率特征，我们上采样的目标是生成两倍于 low 的尺寸（即从 low 的 H,W 上采样到 2H,2W）。所以输出应为 (2, 128, 80, 80)。