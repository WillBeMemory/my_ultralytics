import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== ECA 模块 =====================
class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA) 模块。

    通过自适应一维卷积快速建模通道依赖，避免降维造成的特征损失。
    参数:
        channels (int): 输入特征图的通道数。
        gamma   (int): 控制核大小的超参数，默认 2。
        b       (int): 控制核大小的超参数，默认 1。
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # 自适应计算一维卷积的核大小
        t = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = t if t % 2 else t + 1  # 保证核大小为奇数
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # 全局平均池化 → (B, C, 1, 1)
        y = x.mean(dim=(2, 3), keepdim=True)
        # 转换为序列格式: (B, C, 1) → (B, 1, C)
        y = y.squeeze(-1).transpose(-1, -2)
        # 一维卷积捕捉跨通道交互
        y = self.conv(y)                      # (B, 1, C)
        # 恢复形状并激活
        y = y.transpose(-1, -2).unsqueeze(-1) # (B, C, 1, 1)
        y = self.sigmoid(y)
        # 通道加权
        return x * y


# ===================== 小波融合上采样 (集成 ECA) =====================
class WaveletFusionUp(nn.Module):
    """
    双输入小波融合上采样（轻量版，集成 ECA），输出通道与 high 通道相同，尺寸与 low 相同。
    用于替换 YOLO Neck 中的 “nn.Upsample + Concat” 操作。

    参数:
        low_ch  : low 特征的通道数 (如 P4 的 c4)
        high_ch : high 特征的通道数 (如 P5 的 c5)
        k       : 融合后深度可分离卷积的核大小，默认 3
    """
    def __init__(self, low_ch, high_ch, k=3):
        super().__init__()
        self.low_ch = low_ch
        self.high_ch = high_ch

        # ---- Haar 分解滤波器 (固定) ----
        w_ll = torch.tensor([[1., 1.], [1., 1.]]) / 2.0
        w_lh = torch.tensor([[1., -1.], [1., -1.]]) / 2.0
        w_hl = torch.tensor([[1., 1.], [-1., -1.]]) / 2.0
        w_hh = torch.tensor([[1., -1.], [-1., 1.]]) / 2.0
        haar = torch.stack([w_ll, w_lh, w_hl, w_hh], dim=0).unsqueeze(1)  # (4, 1, 2, 2)
        self.register_buffer('dec_filter', haar.repeat(low_ch, 1, 1, 1))

        # ---- 投影层 (low 子带) ----
        self.proj_low = nn.Sequential(
            nn.Conv2d(low_ch * 4, high_ch * 4, 1, bias=False),
            nn.BatchNorm2d(high_ch * 4),
            ECA(high_ch * 4),           # 添加通道注意力
            nn.SiLU(inplace=True)
        )

        # ---- 投影层 (high 特征) ----
        self.proj_high = nn.Sequential(
            nn.Conv2d(high_ch, high_ch * 4, 1, bias=False),
            nn.BatchNorm2d(high_ch * 4),
            ECA(high_ch * 4),           # 添加通道注意力
            nn.SiLU(inplace=True)
        )

        # ---- 融合微调 (深度可分离卷积) ----
        self.fuse = nn.Sequential(
            nn.Conv2d(high_ch * 4, high_ch * 4, k, padding=k//2, groups=high_ch*4, bias=False),
            nn.BatchNorm2d(high_ch * 4),
            nn.SiLU(inplace=True)
        )

        # ---- 输出微调 (1x1 + ECA) ----
        self.out_conv = nn.Sequential(
            nn.Conv2d(high_ch, high_ch, 1, bias=False),
            ECA(high_ch)                # 输出前增强通道交互
        )

    def forward(self, low, high):
        B, _, H, W = low.shape

        # 1. 对 low 进行 Haar 小波分解 → (B, 4*low_ch, H/2, W/2)
        coeff = F.conv2d(low, self.dec_filter, stride=2, groups=self.low_ch)

        # 2. 投影到相同特征空间
        low_feat = self.proj_low(coeff)      # (B, 4*high_ch, H/2, W/2)
        high_feat = self.proj_high(high)     # (B, 4*high_ch, H/2, W/2)

        # 3. 融合并微调
        fused = low_feat + high_feat
        fused = self.fuse(fused)             # (B, 4*high_ch, H/2, W/2)

        # 4. 重整为 (B, high_ch, 4, H/2, W/2)
        fused = fused.view(B, self.high_ch, 4, H//2, W//2)

        # 5. 逆 Haar 小波重建 (Pixel Shuffle)
        combined = fused.reshape(B * self.high_ch, 4, H//2, W//2)
        up = F.pixel_shuffle(combined, 2)    # (B*high_ch, 1, H, W)
        out = up.reshape(B, self.high_ch, H, W)

        # 6. 最终微调 + ECA
        out = self.out_conv(out)             # (B, high_ch, H, W)

        return out


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 模拟 P4 (low) 和 P5 (high) 特征图
    low  = torch.randn(2, 128, 40, 40).to(device)   # c4 = 128
    high = torch.randn(2, 256, 20, 20).to(device)   # c5 = 256

    model = WaveletFusionUp(low_ch=128, high_ch=256, k=3).to(device)
    out = model(low, high)

    print(f"low  shape: {low.shape}")
    print(f"high shape: {high.shape}")
    print(f"out  shape: {out.shape}")  # 期望 (2, 256, 40, 40)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 简单的 ECA 验证
    eca_test = ECA(64).to(device)
    x_test = torch.randn(2, 64, 16, 16).to(device)
    out_eca = eca_test(x_test)
    print(f"\nECA 测试: 输入 {x_test.shape} -> 输出 {out_eca.shape} (相同)")
    print(f"ECA 参数量: {sum(p.numel() for p in eca_test.parameters()):,}")