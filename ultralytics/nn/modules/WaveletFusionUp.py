import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletFusionUp(nn.Module):
    """
    双输入小波融合上采样（可轻量配置）
    输入：
        low  : 高分辨率特征 (B, C_low, H, W)   —— 作为 LL 子带
        high : 低分辨率特征 (B, C_high, H/2, W/2)
    输出：
        上采样后的特征图 (B, C_high, H, W)

    参数：
        low_ch     : low 特征的通道数
        high_ch    : high 特征的通道数
        k          : 深度可分离卷积的核大小
        compression: 内部通道压缩比。
                     原版性能最优，值越大参数越少。
    """
    def __init__(self, low_ch, high_ch, k=3, compression=1):
        super().__init__()
        self.low_ch = low_ch
        self.high_ch = high_ch
        self.compression = compression

        # 压缩后的内部子带总和通道数（原为 4 * high_ch）
        inner_ch = max(1, (4 * high_ch) // compression)

        # ---- 固定 Haar 分解滤波器 (作用于 low) ----
        w_ll = torch.tensor([[1., 1.], [1., 1.]]) / 2.0
        w_lh = torch.tensor([[1., -1.], [1., -1.]]) / 2.0
        w_hl = torch.tensor([[1., 1.], [-1., -1.]]) / 2.0
        w_hh = torch.tensor([[1., -1.], [-1., 1.]]) / 2.0
        haar = torch.stack([w_ll, w_lh, w_hl, w_hh], dim=0).unsqueeze(1)  # (4,1,2,2)
        self.register_buffer('dec_filter', haar.repeat(low_ch, 1, 1, 1))

        # ---- 可学习投影层 ----
        # 将 low 的 4 个子带压缩到 inner_ch
        self.proj_low = nn.Sequential(
            nn.Conv2d(low_ch * 4, inner_ch, 1, bias=False),
            nn.BatchNorm2d(inner_ch),
            nn.SiLU(inplace=True)
        )
        # 将 high 投影到 inner_ch
        self.proj_high = nn.Sequential(
            nn.Conv2d(high_ch, inner_ch, 1, bias=False),
            nn.BatchNorm2d(inner_ch),
            nn.SiLU(inplace=True)
        )

        # 如果压缩了，需要用 1x1 将 inner_ch 恢复到 4*high_ch（用于逆变换）
        if compression != 1:
            self.expand = nn.Conv2d(inner_ch, 4 * high_ch, 1, bias=False)
        else:
            self.expand = nn.Identity()

        # 融合后的微调：深度可分离卷积，参数量极小
        self.fuse = nn.Sequential(
            nn.Conv2d(inner_ch, inner_ch, k, padding=k//2, groups=inner_ch, bias=False),
            nn.BatchNorm2d(inner_ch),
            nn.SiLU(inplace=True)
        )

    def forward(self, low, high):
        B, _, H, W = low.shape

        # 1. 对 low 进行 Haar 小波分解 → (B, 4*low_ch, H/2, W/2)
        coeff = F.conv2d(low, self.dec_filter, stride=2, groups=self.low_ch)

        # 2. 投影到相同的子带特征空间
        low_feat = self.proj_low(coeff)          # (B, inner_ch, H/2, W/2)
        high_feat = self.proj_high(high)         # (B, inner_ch, H/2, W/2)

        # 3. 融合并微调
        fused = low_feat + high_feat
        fused = self.fuse(fused)                 # (B, inner_ch, H/2, W/2)

        # 4. (可选) 恢复到 4*high_ch 以适配逆 Haar 变换
        fused = self.expand(fused)               # (B, 4*high_ch, H/2, W/2)

        # 5. 重整为 (B, high_ch, 4, H/2, W/2)
        fused = fused.view(B, self.high_ch, 4, H // 2, W // 2)

        # 6. 逆 Haar 小波重建 (Pixel Shuffle)
        combined = fused.reshape(B * self.high_ch, 4, H // 2, W // 2)
        up = F.pixel_shuffle(combined, 2)        # (B*high_ch, 1, H, W)
        out = up.reshape(B, self.high_ch, H, W)

        return out


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 模拟 P4 (low) 和 P5 (high) 特征
    low  = torch.randn(2, 128, 40, 40).to(device)   # P4
    high = torch.randn(2, 256, 20, 20).to(device)   # P5

    for comp in [1, 2, 4]:
        model = WaveletFusionUp(low_ch=128, high_ch=256, k=3, compression=comp).to(device)
        out = model(low, high)
        params = sum(p.numel() for p in model.parameters())
        print(f"compression={comp}: 输出形状 {out.shape}")
        print(f"  参数量: {params:,}")