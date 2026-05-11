import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class SparseSelfAttention(nn.Module):
    """稀疏自注意力模块（余弦注意力变体）"""
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        x = self.norm(x + out)
        return x


class HIPABlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_levels: int = 3,
        threshold_init: float = 0.5,
        use_contrast_norm: bool = True,
        importance_mode: str = 'l2',
        fill_mode: str = 'upsample',   # 'upsample' 或 'constant'
        aggregate: bool = False,        # 是否启用多级重要性聚合
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.use_contrast_norm = use_contrast_norm
        self.importance_mode = importance_mode
        self.fill_mode = fill_mode
        self.aggregate = aggregate

        # 可学习阈值（每个层级独立，但实际仅使用最粗层级的阈值）
        init_logit = np.log(threshold_init / (1.0 - threshold_init + 1e-8))
        self.logit_thresholds = nn.Parameter(torch.full((num_levels,), init_logit))

        # 投影层（用于自注意力序列的特征变换）
        self.proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, out_channels)
        )

        # 上采样模块（仅在 fill_mode='upsample' 时使用）
        if fill_mode == 'upsample':
            self.upsampler = nn.Upsample(scale_factor=None, mode='bilinear', align_corners=False)

    def get_thresholds(self):
        return torch.sigmoid(self.logit_thresholds)

    @staticmethod
    def _deterministic_grid_pool(x, grid_size):
        B, C, H, W = x.shape
        pad_h = (grid_size - H % grid_size) % grid_size
        pad_w = (grid_size - W % grid_size) % grid_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        H_pad, W_pad = H + pad_h, W + pad_w
        kernel_h = H_pad // grid_size
        kernel_w = W_pad // grid_size
        return F.max_pool2d(x, kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))

    def _get_coords(self, B, H, W, grid_h, grid_w, dtype, device):
        y_centers = torch.linspace(0.5 / grid_h, 1 - 0.5 / grid_h, grid_h, device=device, dtype=dtype)
        x_centers = torch.linspace(0.5 / grid_w, 1 - 0.5 / grid_w, grid_w, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(y_centers, x_centers, indexing='ij')
        centers = torch.stack([gx, gy], dim=-1).reshape(-1, 2)
        sizes = torch.tensor([1.0 / grid_w, 1.0 / grid_h], device=device, dtype=dtype).expand(grid_h * grid_w, -1)
        bboxes = torch.cat([centers, sizes], dim=-1)
        return bboxes.unsqueeze(0).expand(B, -1, -1)

    def _fill_mask_constant(self, mask_values, coords, H, W):
        """
        将每个网格的掩膜值填入对应的矩形区域。
        mask_values: (B, N)   软掩膜值 (0~1)
        coords: (B, N, 4)     归一化坐标
        返回: (B, H, W)  单通道掩膜图
        """
        B, N = mask_values.shape
        out = torch.zeros(B, H, W, device=mask_values.device, dtype=mask_values.dtype)
        cx, cy, w, h = coords[..., 0], coords[..., 1], coords[..., 2], coords[..., 3]
        x1 = ((cx - w / 2) * W).long().clamp(0, W - 1)
        y1 = ((cy - h / 2) * H).long().clamp(0, H - 1)
        x2 = ((cx + w / 2) * W).long().clamp(0, W)
        y2 = ((cy + h / 2) * H).long().clamp(0, H)

        for b in range(B):
            valid = (x2[b] > x1[b]) & (y2[b] > y1[b])
            idx = torch.nonzero(valid).squeeze(1)
            if idx.ndim == 0:
                idx = idx.unsqueeze(0)
            for i in idx:
                ya, yb = y1[b, i].item(), y2[b, i].item()
                xa, xb = x1[b, i].item(), x2[b, i].item()
                if xb > xa and yb > ya:
                    out[b, ya:yb, xa:xb] = mask_values[b, i]
        return out

    def _hipa_core(self, x: torch.Tensor, target_mask: Optional[torch.Tensor] = None):
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # 构建金字塔
        pyramid_features, pyramid_coords = [], []
        for level in range(self.num_levels):
            grid_size = 2 ** level
            pooled = self._deterministic_grid_pool(x, grid_size)
            pyramid_features.append(pooled)
            coords = self._get_coords(B, H, W, grid_size, grid_size, dtype, device)
            pyramid_coords.append(coords)

        # 确定最粗层级
        coarse_level = self.num_levels - 1
        thresholds = self.get_thresholds().to(dtype)

        # ---------- 重要性计算 ----------
        if self.aggregate and self.num_levels > 1:
            # ---- 多级聚合模式 ----
            # 1. 计算每层的重要性（L2范数，可选对比度归一化）
            imp_list = []
            for lv in range(1, self.num_levels):   # 跳过全图层 level=0
                feat = pyramid_features[lv]
                _, _, H_l, W_l = feat.shape
                feat_flat = feat.flatten(2).transpose(1, 2)   # (B, N_l, C)
                imp = torch.norm(feat_flat, p=2, dim=-1)      # (B, N_l)

                if self.use_contrast_norm and lv < self.num_levels - 1:
                    parent_feat = pyramid_features[lv + 1]
                    parent_up = F.interpolate(parent_feat, size=(H_l, W_l), mode='nearest')
                    parent_flat = parent_up.flatten(2).transpose(1, 2)
                    parent_norm = torch.norm(parent_flat, p=2, dim=-1)
                    imp = torch.abs(imp - parent_norm)
                imp_list.append(imp.view(B, 1, H_l, W_l))    # (B, 1, H_l, W_l)

            # 2. 将所有比最粗层更细的层上采样到最粗层尺寸并相加
            coarse_feat = pyramid_features[coarse_level]
            _, _, H_c, W_c = coarse_feat.shape
            aggregated = imp_list[-1].clone()   # 最粗层自身 (B,1,H_c,W_c)
            for lv in range(len(imp_list) - 1):  # 不包含最粗层自身
                imp_up = F.interpolate(imp_list[lv], size=(H_c, W_c),
                                       mode='bilinear', align_corners=False)
                aggregated = aggregated + imp_up

            # 展平为序列
            aggregated_flat = aggregated.flatten(2).transpose(1, 2).squeeze(-1)  # (B, N_c)
            importance = aggregated_flat
        else:
            # ---- 原始模式：仅使用最粗层的重要性 ----
            feat = pyramid_features[coarse_level]
            _, _, H_l, W_l = feat.shape
            feat_flat = feat.flatten(2).transpose(1, 2)
            importance = torch.norm(feat_flat, p=2, dim=-1)
            if self.use_contrast_norm and coarse_level < self.num_levels - 1:
                parent_feat = pyramid_features[coarse_level + 1]
                parent_up = F.interpolate(parent_feat, size=(H_l, W_l), mode='nearest')
                parent_flat = parent_up.flatten(2).transpose(1, 2)
                parent_norm = torch.norm(parent_flat, p=2, dim=-1)
                importance = torch.abs(importance - parent_norm)

        # 归一化到 [0,1]
        imp_min = importance.min(dim=1, keepdim=True)[0]
        imp_max = importance.max(dim=1, keepdim=True)[0]
        importance_01 = (importance - imp_min) / (imp_max - imp_min + 1e-8)

        # 目标掩膜（可选）
        coarse_feat = pyramid_features[coarse_level]
        _, _, H_c, W_c = coarse_feat.shape
        if target_mask is not None:
            target_mask = target_mask.to(dtype)
            target_down = F.interpolate(
                target_mask.float().unsqueeze(1), size=(H_c, W_c), mode='nearest'
            ).squeeze(1).flatten(1)
            hard_target = (target_down > 0.2).to(dtype)
        else:
            hard_target = torch.zeros_like(importance)

        # 背景掩膜（可学习阈值 + STE）
        thresh = thresholds[coarse_level]
        bg_mask_hard = (importance_01 >= thresh).to(dtype)
        bg_mask_soft = torch.sigmoid((importance_01 - thresh) * 10.0)
        bg_mask = bg_mask_hard - bg_mask_soft.detach() + bg_mask_soft
        mask = torch.max(hard_target, bg_mask)   # (B, N_c)

        # 掩膜扩展
        final_soft_mask = torch.zeros(B, H, W, device=device, dtype=dtype)
        if self.fill_mode == 'upsample':
            mask_2d = mask.view(B, H_c, W_c)
            self.upsampler.size = (H, W)
            up_mask = self.upsampler(mask_2d.unsqueeze(1)).squeeze(1)
            if up_mask.dtype != dtype:
                up_mask = up_mask.to(dtype)
            final_soft_mask = up_mask
        elif self.fill_mode == 'constant':
            final_soft_mask = self._fill_mask_constant(mask, pyramid_coords[coarse_level], H, W)

        # 自注意力序列收集
        batch_kept_feats = [[] for _ in range(B)]
        batch_kept_coords = [[] for _ in range(B)]
        if self.aggregate or True:  # 保持逻辑一致
            hard_mask = (mask > 0.5).to(dtype)
            for b in range(B):
                keep_idx = torch.nonzero(hard_mask[b], as_tuple=True)[0]
                if len(keep_idx) > 0:
                    selected_feat = pyramid_features[coarse_level].flatten(2).transpose(1, 2)[b, keep_idx, :]
                    selected_coord = pyramid_coords[coarse_level][b, keep_idx, :]
                    selected_feat = self.proj(selected_feat)
                    batch_kept_feats[b].append(selected_feat)
                    batch_kept_coords[b].append(selected_coord)

        # 生成稀疏输出
        if self.training:
            out_sparse = x * final_soft_mask.unsqueeze(1)
        else:
            hard_final = (final_soft_mask > 0.5).to(dtype)
            out_sparse = x * hard_final.unsqueeze(1)

        # 拼接保留的特征和坐标
        for b in range(B):
            if batch_kept_feats[b]:
                batch_kept_feats[b] = torch.cat(batch_kept_feats[b], dim=0)
                batch_kept_coords[b] = torch.cat(batch_kept_coords[b], dim=0)
            else:
                batch_kept_feats[b] = torch.zeros(0, self.out_channels, device=device, dtype=dtype)
                batch_kept_coords[b] = torch.zeros(0, 4, device=device, dtype=dtype)

        return out_sparse, batch_kept_feats, batch_kept_coords

    def forward(self, x: torch.Tensor, target_mask: Optional[torch.Tensor] = None):
        return self._hipa_core(x, target_mask)


class HIPA(nn.Module):
    def __init__(
        self, c1, c2, n=1, num_levels=3,
        threshold_init=0.5,
        use_contrast_norm=True,
        use_self_attn=True,
        importance_mode='l2',
        num_heads=8,
        fill_mode='upsample',
        aggregate=True,         # 新增聚合参数
    ):
        super().__init__()
        self.n = n
        self.use_self_attn = use_self_attn

        self.hipa = HIPABlock(
            in_channels=c1, out_channels=c2,
            num_levels=num_levels,
            threshold_init=threshold_init,
            use_contrast_norm=use_contrast_norm,
            importance_mode=importance_mode,
            fill_mode=fill_mode,
            aggregate=aggregate,
        )

        if use_self_attn:
            self.attn = SparseSelfAttention(embed_dim=c2, num_heads=num_heads)
        else:
            self.attn = nn.Identity()

    def forward(self, x, target_mask=None):
        B, C, H, W = x.shape
        out_sparse, kept_feats, kept_coords = self.hipa(x, target_mask)

        if self.use_self_attn and not isinstance(self.attn, nn.Identity):
            out_attn = torch.zeros(B, C, H, W, device=x.device, dtype=x.dtype)
            for b in range(B):
                feats = kept_feats[b]
                coords = kept_coords[b]
                if feats.shape[0] == 0:
                    continue
                feats_b = feats.to(x.dtype).unsqueeze(0)
                updated = self.attn(feats_b).squeeze(0)
                coords_b = coords.unsqueeze(0)

                smap = torch.zeros(1, C, H, W, device=x.device, dtype=x.dtype)
                cx = (coords_b[..., 0] * W).long().clamp(0, W - 1)
                cy = (coords_b[..., 1] * H).long().clamp(0, H - 1)
                index = (cy * W + cx).unsqueeze(1).expand(-1, C, -1)
                updated = updated.to(x.dtype)
                smap = smap.flatten(2).scatter_(2, index, updated.unsqueeze(0).transpose(1, 2)).view(1, C, H, W)
                out_attn[b] = smap
            return out_attn
        else:
            return out_sparse


# ---------- 简单测试 ----------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(2, 256, 20, 20, device=device)
    model_default = HIPA(256, 256, num_levels=3, threshold_init=0.5,
                         use_self_attn=False, fill_mode='constant', aggregate=False).to(device)
    out1 = model_default(x)
    print(f"Default output shape: {out1.shape}")

    model_agg = HIPA(256, 256, num_levels=3, threshold_init=0.5,
                     use_self_attn=False, fill_mode='constant', aggregate=True).to(device)
    out2 = model_agg(x)
    print(f"Aggregated output shape: {out2.shape}")
    print("Test passed.")