import torch
import torch.nn as nn
import torch.nn.functional as F

from ..conv import Conv


# ================== SoftFill 专用 Bottleneck（残差后激活） ==================
class Bottleneck(nn.Module):
    """Bottleneck with post-activation residual: SiLU is applied AFTER the skip connection.

    Unlike the standard YOLO Bottleneck (which activates before the add and has no
    activation after), this version applies SiLU after the residual addition, which
    is critical for SoftFillEdgeEnhance's feature refinement behavior.
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_, c2, k[1], 1, padding=k[1] // 2, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.act = nn.SiLU(inplace=True)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return self.act(x + out) if self.add else self.act(out)


# ================== 背景填充模块（软填充） ==================
class AdaptiveBackgroundFill(nn.Module):
    def __init__(self, ch, pool_size=3, fill_strength=0.8, bg_thresh_ratio=0.5):
        super().__init__()
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
        # 与 x 同类型的 epsilon
        eps = torch.tensor(1e-6, dtype=x.dtype, device=x.device)
        mean_contrast = local_contrast.mean(dim=[2, 3], keepdim=True) + eps
        # 生成背景掩膜，并强制转为 x 的 dtype
        bg_mask = (local_contrast < self.bg_thresh_ratio * mean_contrast).to(dtype=x.dtype)
        out = x * (1 - self.fill_strength * bg_mask) + baseline * self.fill_strength * bg_mask
        return out


# ================== 通道感知边缘增强（注意力） ==================
class ChannelAwareEdgeEnhance_Attn(nn.Module):
    def __init__(self, ch, pool_size=3, ch_sharp=5.0, ch_thresh=0.5, edge_sharp=5.0, edge_thresh=0.5):
        super().__init__()
        self.ch = ch
        self.pool_size = pool_size
        # 用 register_buffer 存储超参数，随模型移动设备，但不参与梯度
        self.register_buffer('ch_sharp', torch.tensor(ch_sharp))
        self.register_buffer('ch_thresh', torch.tensor(ch_thresh))
        self.register_buffer('edge_sharp', torch.tensor(edge_sharp))
        self.register_buffer('edge_thresh', torch.tensor(edge_thresh))

    def forward(self, x):
        # 动态转换为与输入相同的 dtype，消除 AMP 类型冲突
        dtype = x.dtype
        ch_sharp = self.ch_sharp.to(dtype)
        ch_thresh = self.ch_thresh.to(dtype)
        edge_sharp = self.edge_sharp.to(dtype)
        edge_thresh = self.edge_thresh.to(dtype)

        B, C, H, W = x.shape
        pad = self.pool_size // 2
        x_abs = x.abs()

        # 通道权重（用 view + max 替代 adaptive_max_pool2d，避免确定性警告）
        max_ch = x_abs.view(B, C, -1).max(dim=-1, keepdim=True)[0].view(B, C, 1, 1)
        avg_ch = F.adaptive_avg_pool2d(x_abs, 1)
        diff_ch = max_ch - avg_ch
        ch_weight = torch.sigmoid(ch_sharp * (diff_ch - ch_thresh))

        # 空间边缘权重
        x_spatial = x_abs.mean(dim=1, keepdim=True)
        max_s = F.max_pool2d(x_spatial, self.pool_size, stride=1, padding=pad)
        min_s = -F.max_pool2d(-x_spatial, self.pool_size, stride=1, padding=pad)
        edge = max_s - min_s
        edge_weight = torch.sigmoid(edge_sharp * (edge - edge_thresh))

        out = x * ch_weight
        out = out * (1.0 + edge_weight)
        return out


# ================== 完整模块：SoftFillEdgeEnhance ==================
class SoftFillEdgeEnhance(nn.Module):
    def __init__(self, c1, c2, n=1, pool_size=3,bottleneck_e=0.5,
                 fill_strength=0.8, bg_thresh_ratio=0.5,
                 ch_sharp=5.0, ch_thresh=0.5,
                 edge_sharp=5.0, edge_thresh=0.5,
                  bottleneck_shortcut=True):
        super().__init__()
        self.bg_fill = AdaptiveBackgroundFill(c1, pool_size, fill_strength, bg_thresh_ratio)
        self.attn = ChannelAwareEdgeEnhance_Attn(c1, pool_size, ch_sharp, ch_thresh, edge_sharp, edge_thresh)

        self.bottlenecks = nn.Sequential(*[
            Bottleneck(c1, c1, shortcut=False, e=bottleneck_e)
            for _ in range(n)
        ])

        self.proj = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

    def forward(self, x):
        out = self.bg_fill(x)
        out = self.attn(out)
        out = self.bottlenecks(out)
        out = self.proj(out)
        return out