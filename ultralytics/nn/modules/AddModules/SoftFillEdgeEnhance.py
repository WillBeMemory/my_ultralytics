import torch
import torch.nn as nn
import torch.nn.functional as F


# ================== 标准 Bottleneck ==================
class Bottleneck(nn.Module):
    """标准 Bottleneck：1x1 降维 → 3x3 卷积 → 1x1 升维，可选残差连接"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, k[0], 1, padding=k[0] // 2, bias=False)
        self.cv2 = nn.Conv2d(c_, c2, k[1], 1, padding=k[1] // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.cv1(x)))
        out = self.bn2(self.cv2(out))
        if self.add:
            out = self.act(out + identity)
        else:
            out = self.act(out)
        return out


# ================== 背景填充模块（可学习强度 + 目标保护） ==================
class AdaptiveBackgroundFill(nn.Module):
    def __init__(self, ch, pool_size=3, bg_thresh_ratio=0.5,
                 initial_strength=0.8, protect_target=True, target_thresh=2.0):
        super().__init__()
        self.ch = ch
        self.pool_size = pool_size
        self.bg_thresh_ratio = bg_thresh_ratio
        self.protect_target = protect_target
        self.target_thresh = target_thresh

        # 可学习的填充强度 (1, C, 1, 1)
        self.fill_strength_raw = nn.Parameter(
            torch.full((1, ch, 1, 1), initial_strength)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # 全局最小值作为填充目标
        baseline = x.view(B, C, -1).min(dim=-1)[0].view(B, C, 1, 1)

        # 局部对比度（max - min）
        pad = self.pool_size // 2
        max_s = F.max_pool2d(x, self.pool_size, stride=1, padding=pad)
        min_s = -F.max_pool2d(-x, self.pool_size, stride=1, padding=pad)
        local_contrast = max_s - min_s

        # 动态背景阈值（跟随 x 的 dtype）
        eps = torch.tensor(1e-6, dtype=x.dtype, device=x.device)
        mean_contrast = local_contrast.mean(dim=[2, 3], keepdim=True) + eps
        bg_mask = (local_contrast < self.bg_thresh_ratio * mean_contrast).to(dtype=x.dtype)

        # ---- 目标保护机制 ----
        if self.protect_target:
            # 每个通道的全局最大值和平均值
            max_ch = x.amax(dim=[2, 3], keepdim=True)   # (B, C, 1, 1)
            avg_ch = x.mean(dim=[2, 3], keepdim=True)   # (B, C, 1, 1)
            # 相对峰值指标
            target_score = (max_ch - avg_ch) / (avg_ch + eps)
            # 目标掩膜：峰值足够高的通道视为包含目标，不填充
            target_mask = (target_score > self.target_thresh).to(dtype=x.dtype)
            # 将目标通道的 bg_mask 清零（不填充）
            bg_mask = bg_mask * (1 - target_mask)

        # 可学习的填充强度，通过 sigmoid 约束到 (0,1)
        strength = torch.sigmoid(self.fill_strength_raw)   # (1, C, 1, 1)

        # 软填充
        out = x * (1 - strength * bg_mask) + baseline * strength * bg_mask
        return out

# ================== 通道感知边缘增强（注意力） ==================
class ChannelAwareEdgeEnhance_Attn(nn.Module):
    """通道筛选 + 空间边缘增强，不改变通道数"""
    def __init__(self, ch, pool_size=3, ch_sharp=5.0, ch_thresh=0.5,
                 edge_sharp=5.0, edge_thresh=0.5):
        super().__init__()
        self.ch = ch
        self.pool_size = pool_size
        # 使用 register_buffer 存储固定超参数，AMP 会自动转换 dtype
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

        # 通道权重（用 view+max 替代 adaptive_max_pool2d，避免确定性警告）
        max_ch = x_abs.view(B, C, -1).max(dim=-1, keepdim=True)[0].view(B, C, 1, 1)
        avg_ch = F.adaptive_avg_pool2d(x_abs, 1)
        diff_ch = max_ch - avg_ch
        ch_weight = torch.sigmoid(ch_sharp * (diff_ch - ch_thresh))

        # 空间边缘权重（max - min）
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
    """
    背景软填充（可学习强度 + 目标保护） + 通道‑空间注意力增强 + Bottleneck 精炼

    参数顺序（YOLO 风格）：
    c1, c2, n, pool_size, bg_thresh_ratio, initial_strength, protect_target, target_thresh,
    ch_sharp, ch_thresh, edge_sharp, edge_thresh, bottleneck_e, bottleneck_shortcut
    """
    def __init__(self, c1, c2, n=1, pool_size=3,bottleneck_e=0.5,
                 bg_thresh_ratio=0.3, initial_strength=0.5,
                 protect_target=True, target_thresh=2.0,
                 ch_sharp=10.0, ch_thresh=0.5,
                 edge_sharp=10.0, edge_thresh=0.8,
                  bottleneck_shortcut=True):
        super().__init__()
        self.bg_fill = AdaptiveBackgroundFill(
            c1, pool_size,
            bg_thresh_ratio=bg_thresh_ratio,
            initial_strength=initial_strength,
            protect_target=protect_target,
            target_thresh=target_thresh
        )
        self.attn = ChannelAwareEdgeEnhance_Attn(c1, pool_size, ch_sharp, ch_thresh, edge_sharp, edge_thresh)

        # n 个串联的 Bottleneck
        self.bottlenecks = nn.Sequential(*[
            Bottleneck(c1, c1, shortcut=bottleneck_shortcut, e=bottleneck_e)
            for _ in range(n)
        ])

        # 输出通道对齐（当 c2 != c1 时）
        self.proj = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

    def forward(self, x):
        # 对齐到 Bottleneck 权重的 dtype（AMP 下为 float16），杜绝类型冲突
        if len(self.bottlenecks) > 0:
            target_dtype = self.bottlenecks[0].cv1.weight.dtype
        else:
            target_dtype = x.dtype
        x = x.to(target_dtype)

        out = self.bg_fill(x)        # 1. 背景填充（带目标保护）
        out = self.attn(out)         # 2. 注意力增强（通道筛选+边缘增强）
        out = self.bottlenecks(out)  # 3. 特征精炼
        out = self.proj(out)         # 4. 通道对齐
        return out