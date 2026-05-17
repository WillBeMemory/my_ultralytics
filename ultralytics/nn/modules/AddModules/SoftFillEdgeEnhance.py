import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv


# ================== 密集深度可分离卷积 ==================
class DenseDWConv(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.depthwise = nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False)
        self.dw_bn = nn.BatchNorm2d(c1)
        self.pointwise = nn.Conv2d(c1, c2, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        identity = x
        out = self.act(self.dw_bn(self.depthwise(x)))
        out = self.pw_bn(self.pointwise(out))
        if self.add:
            out = self.act(out + identity)
        else:
            out = self.act(out)
        return out


# ================== 密集 Bottleneck ==================
class DenseBottleneck(nn.Module):
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


# ================== 稀疏 Bottleneck ==================
class SparseBottleneck(nn.Module):
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


# ================== 背景填充模块（固定填充强度） ==================
class AdaptiveBackgroundFill(nn.Module):
    def __init__(self, ch, pool_size=3, bg_thresh_ratio=0.5, fill_strength=0.8):
        super().__init__()
        self.ch = ch
        self.pool_size = pool_size
        self.bg_thresh_ratio = bg_thresh_ratio
        self.fill_strength = fill_strength

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
        return out, edge_weight


# ================== 双分支稀疏块（内部动态生成稀疏张量） ==================
class DualPathSparseBlock(nn.Module):
    """密集深度可分离卷积 + 稀疏 Bottleneck 并行，融合后残差连接"""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        self.dense_dwconv = DenseDWConv(c1, c2, shortcut=False)
        self.sparse_bottleneck = SparseBottleneck(c1, c2, shortcut=False, e=e)
        self.act = nn.SiLU(inplace=True)
        self.add = shortcut and c1 == c2

    def forward(self, dense_input, fg_mask):
        # 密集分支
        out_dense = self.dense_dwconv(dense_input)

        # 稀疏分支：基于硬掩膜生成稀疏张量
        B, C, H, W = dense_input.shape
        active_mask = fg_mask.squeeze(1).bool()
        batch_idx, spatial_y, spatial_x = torch.where(active_mask)
        if batch_idx.numel() == 0:
            out_sparse = torch.zeros_like(dense_input)
        else:
            features = dense_input[batch_idx, :, spatial_y, spatial_x]
            indices = torch.stack([batch_idx, spatial_y, spatial_x], dim=1).int()
            sparse_input = spconv.SparseConvTensor(features, indices, (H, W), B)
            sparse_out = self.sparse_bottleneck(sparse_input)
            out_sparse = sparse_out.dense()

        out = out_dense + out_sparse
        if self.add:
            out = self.act(out + dense_input)
        else:
            out = self.act(out)
        return out


# ================== 完整模块：SoftFillEdgeEnhance（优化掩膜，无密集残差） ==================
class SoftFillEdgeEnhance(nn.Module):
    def __init__(self, c1, c2, n=1, pool_size=3,
                 bg_thresh_ratio=0.5, fill_strength=0.8,
                 ch_sharp=5.0, ch_thresh=0.5,
                 edge_sharp=5.0, edge_thresh=0.5,
                 bottleneck_e=0.5, bottleneck_shortcut=True,
                 mask_alpha=0.7):
        super().__init__()
        self.bg_fill = AdaptiveBackgroundFill(c1, pool_size, bg_thresh_ratio, fill_strength)
        self.attn = ChannelAwareEdgeEnhance_Attn(c1, pool_size, ch_sharp, ch_thresh, edge_sharp, edge_thresh)
        self.mask_alpha = mask_alpha

        # 可学习掩膜精修：1×1 卷积 + Sigmoid
        self.mask_refiner = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 密集精炼（CPU 回退用）
        self.dense_bottlenecks = nn.Sequential(*[
            DenseBottleneck(c1, c1, shortcut=bottleneck_shortcut, e=bottleneck_e)
            for _ in range(n)
        ])
        self.dense_dwconv = nn.Sequential(*[
            DenseDWConv(c1, c1, shortcut=True) for _ in range(n)
        ])

        # 稀疏双分支块（ModuleList，非 Sequential，因为需要传入额外参数）
        self.dual_sparse_blocks = nn.ModuleList([
            DualPathSparseBlock(c1, c1, shortcut=bottleneck_shortcut, e=bottleneck_e)
            for _ in range(n)
        ])

        self.proj = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

    def forward(self, x):
        # 对齐 dtype（AMP 安全）
        target_dtype = self.dense_bottlenecks[0].cv1.weight.dtype
        x = x.to(target_dtype)

        # 1. 背景填充
        filled, bg_mask = self.bg_fill(x)

        # 2. 注意力增强（获得边缘权重）
        enhanced, edge_w = self.attn(filled)

        # 3. 构建优化掩膜
        fg_prob = 1.0 - bg_mask                     # 逐通道前景概率 (B, C, H, W)
        fg_max = fg_prob.max(dim=1, keepdim=True)[0] # 最大值 (B, 1, H, W)
        combined_raw = self.mask_alpha * fg_max + (1.0 - self.mask_alpha) * edge_w

        # 动态阈值：根据全图平均对比度自适应调整
        thresh = combined_raw.mean(dim=[2, 3], keepdim=True).clamp(0.3, 0.7)
        hard_mask = (combined_raw > thresh).to(combined_raw.dtype)

        # 平滑：最大池化膨胀 + 再阈值化，填补空洞、消除孤点
        hard_mask = F.max_pool2d(hard_mask, kernel_size=3, stride=1, padding=1)
        hard_mask = (hard_mask > 0.5).to(combined_raw.dtype)

        # 可学习精修：输入原始组合图，输出精修后的软掩膜
        refined_mask = self.mask_refiner(combined_raw)  # (B, 1, H, W)，值域 (0,1)

        # 最终掩膜：以 refined_mask 为主，但在 hard_mask 确定的强背景区域大幅压低
        # 这样既保留了可学习性，又利用了物理先验
        fg_mask = refined_mask * hard_mask + refined_mask * (1 - hard_mask) * 0.1
        # 再硬阈值化，用于生成稀疏索引（可微性稍差，但训练早期可用软掩膜替代，这里保持硬阈值）
        fg_mask_hard = (fg_mask > 0.5).to(combined_raw.dtype)

        # 4. 根据设备选择路径
        if x.device.type == 'cuda':
            out = enhanced
            for block in self.dual_sparse_blocks:
                out = block(out, fg_mask_hard)   # 传入硬掩膜生成稀疏张量
        else:
            # CPU 回退：密集 Bottleneck + 密集 DWConv
            out = self.dense_bottlenecks(enhanced)
            out = self.dense_dwconv(out)

        # 5. 输出投影
        out = self.proj(out)
        return out