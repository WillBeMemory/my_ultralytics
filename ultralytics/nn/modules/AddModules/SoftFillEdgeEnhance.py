import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv


# ================== 稀疏 Bottleneck ==================
class SparseBottleneck(nn.Module):
    """稀疏 Bottleneck，仅对激活点（前景）进行卷积，保持激活点数不变"""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = spconv.SubMConv2d(c1, c_, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(c_)
        self.cv2 = spconv.SubMConv2d(c_, c_, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(c_)
        self.cv3 = spconv.SubMConv2d(c_, c2, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(c2)
        self.act = nn.SiLU(inplace=True)
        self.add = shortcut and c1 == c2

    def forward(self, sparse_tensor):
        out = self.cv1(sparse_tensor)
        out = out.replace_feature(self.act(self.bn1(out.features)))

        out = self.cv2(out)
        out = out.replace_feature(self.act(self.bn2(out.features)))

        out = self.cv3(out)
        out = out.replace_feature(self.bn3(out.features))

        if self.add:
            out = out.replace_feature(self.act(out.features + sparse_tensor.features))
        else:
            out = out.replace_feature(self.act(out.features))
        return out


# ================== 背景填充模块（无目标保护） ==================
class AdaptiveBackgroundFill(nn.Module):
    """
    自适应背景填充，无目标保护。
    - fill_strength_raw: 可学习的每通道填充强度（sigmoid 约束）
    - bg_thresh_ratio: 背景判定阈值（局部对比度低于该比例*均值 → 背景）
    """
    def __init__(self, ch, pool_size=3, bg_thresh_ratio=0.5, initial_strength=0.8):
        super().__init__()
        self.ch = ch
        self.pool_size = pool_size
        self.bg_thresh_ratio = bg_thresh_ratio

        # 可学习的填充强度 (1, C, 1, 1)
        self.fill_strength_raw = nn.Parameter(
            torch.full((1, ch, 1, 1), initial_strength)
        )

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

        strength = torch.sigmoid(self.fill_strength_raw)

        out = x * (1 - strength * bg_mask) + baseline * strength * bg_mask
        return out, bg_mask


# ================== 通道感知边缘增强（注意力） ==================
class ChannelAwareEdgeEnhance_Attn(nn.Module):
    def __init__(self, ch, pool_size=3, ch_sharp=5.0, ch_thresh=0.5,
                 edge_sharp=5.0, edge_thresh=0.5):
        super().__init__()
        self.ch = ch
        self.pool_size = pool_size
        self.register_buffer('ch_sharp', torch.tensor(ch_sharp))
        self.register_buffer('ch_thresh', torch.tensor(ch_thresh))
        self.register_buffer('edge_sharp', torch.tensor(edge_sharp))
        self.register_buffer('edge_thresh', torch.tensor(edge_thresh))

    def forward(self, x):
        dtype = x.dtype
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

        out = x * ch_weight
        out = out * (1.0 + edge_weight)
        return out


# ================== 完整模块：SoftFillEdgeEnhance（稀疏精炼，无目标保护） ==================
class SoftFillEdgeEnhance(nn.Module):
    def __init__(self, c1, c2, n=1, pool_size=3,
                 bg_thresh_ratio=0.5, initial_strength=0.8,
                 ch_sharp=5.0, ch_thresh=0.5,
                 edge_sharp=5.0, edge_thresh=0.5,
                 bottleneck_e=0.5, bottleneck_shortcut=True):
        super().__init__()
        self.bg_fill = AdaptiveBackgroundFill(
            c1, pool_size,
            bg_thresh_ratio=bg_thresh_ratio,
            initial_strength=initial_strength
        )
        self.attn = ChannelAwareEdgeEnhance_Attn(c1, pool_size, ch_sharp, ch_thresh, edge_sharp, edge_thresh)

        self.bottlenecks = nn.Sequential(*[
            SparseBottleneck(c1, c1, shortcut=bottleneck_shortcut, e=bottleneck_e)
            for _ in range(n)
        ])

        self.proj = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

    def forward(self, x):
        if len(self.bottlenecks) > 0:
            target_dtype = self.bottlenecks[0].cv1.weight.dtype
        else:
            target_dtype = x.dtype
        x = x.to(target_dtype)

        filled, bg_mask = self.bg_fill(x)
        enhanced = self.attn(filled)

        bg_mask_mean = bg_mask.mean(dim=1, keepdim=True)
        fg_mask = (bg_mask_mean < 0.5).to(dtype=enhanced.dtype)

        B, C, H, W = enhanced.shape
        active_mask = fg_mask.squeeze(1).bool()
        batch_idx, spatial_y, spatial_x = torch.where(active_mask)
        features = enhanced[batch_idx, :, spatial_y, spatial_x]
        indices = torch.stack([batch_idx, spatial_y, spatial_x], dim=1).int()
        sparse_input = spconv.SparseConvTensor(features, indices, (H, W), B)

        sparse_output = self.bottlenecks(sparse_input)
        out = sparse_output.dense()
        out = self.proj(out)
        return out