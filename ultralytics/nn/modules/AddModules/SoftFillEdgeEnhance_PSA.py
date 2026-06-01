import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv, PSABlock


# ================== 原有子模块（保持不变） ==================
class AdaptiveBackgroundFill(nn.Module):
    def __init__(self, ch, pool_size=3, fill_strength=0.8, bg_thresh_ratio=0.5):
        super().__init__()
        assert pool_size % 2 == 1, "pool_size must be odd"
        self.ch = ch
        self.pool_size = pool_size
        self.fill_strength = fill_strength
        self.bg_thresh_ratio = bg_thresh_ratio

    def forward(self, x):
        B, C, H, W = x.shape
        baseline = x.view(B, C, -1).min(dim=-1)[0].view(B, C, 1, 1)
        pad = self.pool_size // 2
        max_s = F.max_pool2d(x, self.pool_size, stride=1, padding=pad)
        min_s = -F.max_pool2d(-x, self.pool_size, stride=1, padding=pad)
        local_contrast = max_s - min_s
        eps = torch.tensor(1e-6, dtype=x.dtype, device=x.device)
        mean_contrast = local_contrast.mean(dim=[2, 3], keepdim=True) + eps
        bg_mask = (local_contrast < self.bg_thresh_ratio * mean_contrast).to(dtype=x.dtype)
        out = x * (1 - self.fill_strength * bg_mask) + baseline * self.fill_strength * bg_mask
        return out


class ChannelAwareEdgeEnhance_Attn(nn.Module):
    def __init__(self, ch, pool_size=3, ch_sharp=5.0, ch_thresh=0.5, edge_sharp=5.0, edge_thresh=0.5):
        super().__init__()
        assert pool_size % 2 == 1, "pool_size must be odd"
        self.ch = ch
        self.pool_size = pool_size
        self.register_buffer('ch_sharp', torch.tensor(ch_sharp))
        self.register_buffer('ch_thresh', torch.tensor(ch_thresh))
        self.register_buffer('edge_sharp', torch.tensor(edge_sharp))
        self.register_buffer('edge_thresh', torch.tensor(edge_thresh))

    def forward(self, x):
        dtype = x.dtype
        device = x.device
        ch_sharp = self.ch_sharp.to(dtype)
        ch_thresh = self.ch_thresh.to(dtype)
        edge_sharp = self.edge_sharp.to(dtype)
        edge_thresh = self.edge_thresh.to(dtype)

        B, C, H, W = x.shape
        pad = self.pool_size // 2
        x_abs = x.abs()

        max_ch = x_abs.view(B, C, -1).max(dim=-1, keepdim=True)[0].view(B, C, 1, 1)
        avg_ch = F.adaptive_avg_pool2d(x_abs, 1)
        diff_ch = max_ch - avg_ch
        ch_weight = torch.sigmoid(ch_sharp * (diff_ch - ch_thresh))

        x_spatial = x_abs.mean(dim=1, keepdim=True)
        max_s = F.max_pool2d(x_spatial, self.pool_size, stride=1, padding=pad)
        min_s = -F.max_pool2d(-x_spatial, self.pool_size, stride=1, padding=pad)
        edge = max_s - min_s
        edge_weight = torch.sigmoid(edge_sharp * (edge - edge_thresh))

        one = torch.tensor(1.0, dtype=dtype, device=device)
        out = x * ch_weight
        out = out * (one + edge_weight)
        return out


# ================== 结合模块（已移除 Bottleneck） ==================
class SoftFillEdgeEnhance_PSA(nn.Module):
    """
    背景填充 + 边缘增强 + C2PSA 注意力（无 Bottleneck）。
    参数兼容 C3k2 / C2PSA 接口。
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,                    # PSABlock 个数
        e: float = 0.5,                # 隐藏通道扩展比
        pool_size: int = 3,
        fill_strength: float = 0.8,
        bg_thresh_ratio: float = 0.5,
        ch_sharp: float = 5.0,
        ch_thresh: float = 0.5,
        edge_sharp: float = 5.0,
        edge_thresh: float = 0.5,
        psablock_shortcut: bool = True,
        attn_ratio: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.c = int(c2 * e)          # PSA 隐藏通道数

        # 1. 背景填充 + 边缘增强
        self.bg_fill = AdaptiveBackgroundFill(c1, pool_size, fill_strength, bg_thresh_ratio)
        self.edge_attn = ChannelAwareEdgeEnhance_Attn(c1, pool_size, ch_sharp, ch_thresh, edge_sharp, edge_thresh)

        # 2. PSA 注意力分支
        self.cv1 = Conv(c1, 2 * self.c, 1)                # 降维
        self.psa_blocks = nn.Sequential(*[
            PSABlock(self.c, attn_ratio=attn_ratio, num_heads=max(self.c // 64, 1), shortcut=psablock_shortcut)
            for _ in range(n)
        ])
        self.cv2 = Conv(2 * self.c, c2, 1)                # 恢复通道

    def forward(self, x):
        # 背景填充 + 边缘增强
        x_enh = self.bg_fill(x)
        x_enh = self.edge_attn(x_enh)         # (B, c1, H, W)

        # 通道压缩 + PSA 注意力
        a, b = self.cv1(x_enh).split((self.c, self.c), dim=1)
        b = self.psa_blocks(b)
        out = self.cv2(torch.cat((a, b), dim=1))

        return out


# ================== 简单测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    x = torch.randn(2, 128, 80, 80).to(device)

    model = SoftFillEdgeEnhance_PSA(
        c1=128, c2=256, n=2, e=0.5,
        fill_strength=0.5, bg_thresh_ratio=0.5,
        ch_sharp=5.0, ch_thresh=0.5,
        edge_sharp=5.0, edge_thresh=0.5,
        psablock_shortcut=True
    ).to(device)
    model.train()

    y = model(x)
    print(f"Input: {x.shape} → Output: {y.shape} (expected [2,256,80,80])")

    loss = y.mean()
    loss.backward()
    print("Gradients OK")