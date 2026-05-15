import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv


# ================== 密集 Bottleneck（CPU 回退 & 对照用） ==================
class DenseBottleneck(nn.Module):
    """标准 Bottleneck，用于 CPU 推理或密集路径"""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        self.cv2 = nn.Conv2d(c_, c_, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_)
        self.cv3 = nn.Conv2d(c_, c2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.cv1(x)))
        out = self.act(self.bn2(self.cv2(out)))
        out = self.bn3(self.cv3(out))
        if self.add:
            out = self.act(out + identity)
        else:
            out = self.act(out)
        return out


# ================== 稀疏 Bottleneck（GPU 加速） ==================
class SparseBottleneck(nn.Module):
    """稀疏 Bottleneck，仅对激活点进行卷积，保持激活点数不变"""
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
    def __init__(self, ch, pool_size=3, bg_thresh_ratio=0.5, initial_strength=0.8):
        super().__init__()
        self.ch = ch
        self.pool_size = pool_size
        self.bg_thresh_ratio = bg_thresh_ratio
        self.fill_strength_raw = nn.Parameter(torch.full((1, ch, 1, 1), initial_strength))

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


# ================== 完整模块：SoftFillEdgeEnhance（稀疏 + CPU回退） ==================
class SoftFillEdgeEnhance(nn.Module):
    def __init__(self, c1, c2, n=1, pool_size=3,
                 bg_thresh_ratio=0.3, initial_strength=0.5,
                 ch_sharp=3.0, ch_thresh=0.3,
                 edge_sharp=5.0, edge_thresh=0.5,
                 bottleneck_e=0.5, bottleneck_shortcut=True):
        super().__init__()
        self.bg_fill = AdaptiveBackgroundFill(c1, pool_size, bg_thresh_ratio, initial_strength)
        self.attn = ChannelAwareEdgeEnhance_Attn(c1, pool_size, ch_sharp, ch_thresh, edge_sharp, edge_thresh)

        # 同时构建密集和稀疏 Bottleneck
        self.dense_bottlenecks = nn.Sequential(*[
            DenseBottleneck(c1, c1, shortcut=bottleneck_shortcut, e=bottleneck_e)
            for _ in range(n)
        ])
        self.sparse_bottlenecks = nn.Sequential(*[
            SparseBottleneck(c1, c1, shortcut=bottleneck_shortcut, e=bottleneck_e)
            for _ in range(n)
        ])

        self.proj = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

    def forward(self, x):
        # 对齐 dtype（AMP 安全）
        target_dtype = self.dense_bottlenecks[0].cv1.weight.dtype
        x = x.to(target_dtype)

        # 1. 背景填充
        filled, bg_mask = self.bg_fill(x)
        # 2. 注意力增强
        enhanced = self.attn(filled)

        # 3. 判断是否使用稀疏路径：要求 CUDA 且至少有一个激活点
        use_sparse = (x.device.type == 'cuda')
        if use_sparse:
            # 生成前景掩膜
            bg_mask_mean = bg_mask.mean(dim=1, keepdim=True)
            fg_mask = (bg_mask_mean < 0.5).to(dtype=enhanced.dtype)
            B, C, H, W = enhanced.shape
            active_mask = fg_mask.squeeze(1).bool()
            batch_idx, spatial_y, spatial_x = torch.where(active_mask)
            if batch_idx.numel() == 0:  # 无激活点，直接返回零张量
                out = torch.zeros_like(enhanced)
            else:
                features = enhanced[batch_idx, :, spatial_y, spatial_x]
                indices = torch.stack([batch_idx, spatial_y, spatial_x], dim=1).int()
                sparse_input = spconv.SparseConvTensor(features, indices, (H, W), B)
                sparse_output = self.sparse_bottlenecks(sparse_input)
                out = sparse_output.dense()
        else:
            # CPU 环境或回退：使用密集 Bottleneck
            out = self.dense_bottlenecks(enhanced)

        # 4. 输出投影
        out = self.proj(out)
        return out