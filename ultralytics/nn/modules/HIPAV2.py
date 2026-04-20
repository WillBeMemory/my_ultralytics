# ============================================
# File: ultralytics/nn/modules/HIPAV2.py
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

__all__ = ["HIPAV2"]


class SparseSelfAttention(nn.Module):
    """单层多头自注意力 + 残差 + LayerNorm"""
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        return x


class HIPABlock(nn.Module):
    """
    Hierarchical Importance Propagation Attention Block
    支持局部网格内自注意力（可选）+ L2对比度筛选 + 全局稀疏自注意力。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_levels: int = 3,
        keep_ratio: float = 0.3,
        min_keeps: int = 8,
        use_contrast_norm: bool = True,
        fill_mode: str = 'constant',
        local_attn: bool = False,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.keep_ratio = keep_ratio
        self.min_keeps = min_keeps
        self.use_contrast_norm = use_contrast_norm
        self.fill_mode = fill_mode
        self.local_attn = local_attn

        self.proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, out_channels)
        )

        if local_attn:
            self.local_attention = SparseSelfAttention(out_channels, num_heads, attn_dropout)

    @staticmethod
    def _deterministic_grid_pool(x: torch.Tensor, grid_size: int) -> torch.Tensor:
        B, C, H, W = x.shape
        pad_h = (grid_size - H % grid_size) % grid_size
        pad_w = (grid_size - W % grid_size) % grid_size
        if pad_h > 0 or pad_w > 0:
            # 改回反射填充（接受非确定性警告以降低 FLOPs）
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        H_pad, W_pad = H + pad_h, W + pad_w
        kernel_h = H_pad // grid_size
        kernel_w = W_pad // grid_size
        return F.max_pool2d(x, kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))

    def _get_coords(self, B: int, H: int, W: int, grid_h: int, grid_w: int,
                    dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        y_centers = torch.linspace(0.5 / grid_h, 1 - 0.5 / grid_h, grid_h, device=device, dtype=dtype)
        x_centers = torch.linspace(0.5 / grid_w, 1 - 0.5 / grid_w, grid_w, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(y_centers, x_centers, indexing='ij')
        centers = torch.stack([gx, gy], dim=-1).reshape(-1, 2)
        sizes = torch.tensor([1.0 / grid_w, 1.0 / grid_h], device=device, dtype=dtype).expand(grid_h * grid_w, -1)
        bboxes = torch.cat([centers, sizes], dim=-1)
        return bboxes.unsqueeze(0).expand(B, -1, -1)

    def _apply_local_attn(self, feat: torch.Tensor, level: int) -> torch.Tensor:
        B, C, H_l, W_l = feat.shape
        feat_flat = feat.flatten(2).transpose(1, 2)
        if feat_flat.size(1) > 1:
            feat_flat = self.local_attention(feat_flat)
        return feat_flat.transpose(1, 2).view(B, C, H_l, W_l)

    def _fill_2d(self, features: torch.Tensor, coords: torch.Tensor,
                 H: int, W: int) -> torch.Tensor:
        B, K, C = features.shape
        dtype = features.dtype
        device = features.device
        out = torch.zeros(B, self.out_channels, H, W, device=device, dtype=dtype)

        if self.fill_mode == 'center':
            cx = (coords[..., 0] * W).long().clamp(0, W - 1)
            cy = (coords[..., 1] * H).long().clamp(0, H - 1)
            src = features.transpose(1, 2)
            index = (cy.unsqueeze(1) * W + cx.unsqueeze(1)).expand(-1, self.out_channels, -1)
            out = out.flatten(2).scatter_(2, index, src).view(B, self.out_channels, H, W)
        else:
            for b in range(B):
                for i in range(K):
                    cx, cy, w, h = coords[b, i]
                    x1 = int((cx - w / 2) * W)
                    y1 = int((cy - h / 2) * H)
                    x2 = int((cx + w / 2) * W)
                    y2 = int((cy + h / 2) * H)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W, x2), min(H, y2)
                    if x2 > x1 and y2 > y1:
                        feat = features[b, i].view(self.out_channels, 1, 1).expand(-1, y2 - y1, x2 - x1)
                        out[b, :, y1:y2, x1:x2] = feat
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        assert C == self.in_channels, f"Input channels {C} != in_channels {self.in_channels}"
        dtype = x.dtype
        device = x.device

        # ----- 1. 构建特征金字塔 -----
        pyramid_features = []
        pyramid_coords = []

        for level in range(self.num_levels):
            grid_size = 2 ** level
            grid_h = grid_w = grid_size

            pooled = self._deterministic_grid_pool(x, grid_size)
            if self.local_attn:
                pooled = self._apply_local_attn(pooled, level)

            pyramid_features.append(pooled)
            coords = self._get_coords(B, H, W, grid_h, grid_w, dtype, device)
            pyramid_coords.append(coords)

        # ----- 2. 自底向上重要性筛选（使用 L2 范数对比度）-----
        kept_coords_list = []
        kept_features_list = []

        for level in reversed(range(self.num_levels)):
            feat = pyramid_features[level]
            B, C_l, H_l, W_l = feat.shape
            N_l = H_l * W_l
            feat_flat = feat.flatten(2).transpose(1, 2)  # (B, N_l, C_l)

            # L2 范数重要性计算
            l2_norm = torch.norm(feat_flat, p=2, dim=-1)  # (B, N_l)

            if self.use_contrast_norm and level < self.num_levels - 1:
                parent_feat = pyramid_features[level + 1]
                parent_feat_up = F.interpolate(parent_feat, size=(H_l, W_l), mode='nearest')
                parent_flat = parent_feat_up.flatten(2).transpose(1, 2)
                parent_l2 = torch.norm(parent_flat, p=2, dim=-1)
                importance = torch.abs(l2_norm - parent_l2)
            else:
                importance = l2_norm

            keep_num = max(self.min_keeps, int(N_l * self.keep_ratio))
            keep_num = min(keep_num, N_l)

            topk_importance, topk_indices = torch.topk(importance, k=keep_num, dim=-1)

            kept_feat = torch.gather(feat_flat, 1, topk_indices.unsqueeze(-1).expand(-1, -1, C_l))
            kept_feat = self.proj(kept_feat)

            kept_coords = torch.gather(pyramid_coords[level], 1,
                                       topk_indices.unsqueeze(-1).expand(-1, -1, 4))

            kept_coords_list.insert(0, kept_coords)
            kept_features_list.insert(0, kept_feat)

        sparse_seq = torch.cat(kept_features_list, dim=1)
        all_coords = torch.cat(kept_coords_list, dim=1)

        # ----- 3. 生成二维稀疏图 -----
        out_sparse = self._fill_2d(sparse_seq, all_coords, H, W)
        total_kept = sparse_seq.size(1)
        sparsity = torch.tensor(total_kept / (H * W), device=device, dtype=dtype).mean()

        return out_sparse, sparse_seq, all_coords, sparsity


class _HIPASingle(nn.Module):
    """单次 HIPA + 可选全局自注意力"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_levels: int = 3,
        keep_ratio: float = 0.3,
        min_keeps: int = 8,
        use_contrast_norm: bool = True,
        num_heads: int = 8,
        fill_mode: str = 'constant',
        use_global_attn: bool = True,
        local_attn: bool = False,
        attn_dropout: float = 0.0,
        fusion_alpha: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_global_attn = use_global_attn
        self.fill_mode = fill_mode
        self.fusion_alpha = fusion_alpha

        self.hipa = HIPABlock(
            in_channels=in_channels,
            out_channels=out_channels,
            num_levels=num_levels,
            keep_ratio=keep_ratio,
            min_keeps=min_keeps,
            use_contrast_norm=use_contrast_norm,
            fill_mode=fill_mode,
            local_attn=local_attn,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
        )

        if use_global_attn:
            self.global_attn = SparseSelfAttention(embed_dim=out_channels, num_heads=num_heads, dropout=attn_dropout)
        else:
            self.global_attn = nn.Identity()

        if in_channels != out_channels:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = nn.Identity()

        # if use_global_attn:
        #     self.fusion_weight = nn.Parameter(torch.tensor(fusion_alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_sparse_raw, sparse_seq, all_coords, _ = self.hipa(x)
        H, W = x.shape[2:]

        if self.use_global_attn:
            sparse_seq_updated = self.global_attn(sparse_seq)
            sparse_seq = sparse_seq + sparse_seq_updated
            out_sparse = self.hipa._fill_2d(sparse_seq, all_coords, H, W)
        else:
            out_sparse = out_sparse_raw

        residual = self.residual_proj(x)
        return residual + out_sparse


class HIPAV2(nn.Module):
    """
    HIPAV2 模块（支持重复 n 次），输出二维稀疏特征图。
    重要性度量：L2 范数对比度（当前版本）

    参数顺序（YOLO 配置文件 args 列表）：
        in_channels (int)          : 输入通道数
        out_channels (int)         : 输出通道数
        n (int)                    : 模块重复次数，默认 1
        num_levels (int)           : 金字塔层级数，默认 3
        keep_ratio (float)         : 每层保留比例，默认 0.3
        min_keeps (int)            : 每层最少保留区域数，默认 8
        use_contrast_norm (bool)   : 是否使用对比度归一化，默认 True
        num_heads (int)            : 自注意力头数，默认 8
        fill_mode (str)            : 填充模式，'constant' 或 'center'，默认 'constant'
        use_global_attn (bool)     : 是否启用全局自注意力，默认 True
        local_attn (bool)          : 是否启用网格内局部自注意力，默认 False
        attn_dropout (float)       : 注意力 dropout 率，默认 0.0
        fusion_alpha (float)       : 初始融合权重，默认 0.5
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        num_levels: int = 3,
        keep_ratio: float = 0.3,
        min_keeps: int = 8,
        use_contrast_norm: bool = True,
        num_heads: int = 8,
        fill_mode: str = 'constant',
        use_global_attn: bool = True,
        local_attn: bool = False,
        attn_dropout: float = 0.0,
        fusion_alpha: float = 0.5,
    ):
        super().__init__()
        self.n = n
        self.in_channels = in_channels
        self.out_channels = out_channels

        blocks = []
        for i in range(n):
            in_ch = in_channels if i == 0 else out_channels
            blocks.append(
                _HIPASingle(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    num_levels=num_levels,
                    keep_ratio=keep_ratio,
                    min_keeps=min_keeps,
                    use_contrast_norm=use_contrast_norm,
                    num_heads=num_heads,
                    fill_mode=fill_mode,
                    use_global_attn=use_global_attn,
                    local_attn=local_attn,
                    attn_dropout=attn_dropout,
                    fusion_alpha=fusion_alpha,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


# ============================================
# 测试代码（本地验证）
# ============================================
if __name__ == "__main__":
    dummy = torch.randn(2, 256, 20, 20, device='cuda', dtype=torch.float16)
    model = HIPAV2(
        in_channels=256,
        out_channels=512,
        n=1,
        num_levels=3,
        keep_ratio=0.2,
        min_keeps=4,
        use_contrast_norm=True,
        num_heads=4,
        fill_mode='center',
        use_global_attn=True,
        local_attn=False,
        attn_dropout=0.1,
        fusion_alpha=0.5,
    ).cuda().half()
    out = model(dummy)
    print(f"Input: {dummy.shape}, {dummy.dtype}")
    print(f"Output: {out.shape}, {out.dtype}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")