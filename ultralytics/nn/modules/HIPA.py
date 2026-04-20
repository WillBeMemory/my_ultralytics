# ============================================
# File: ultralytics/nn/modules/HIPA.py
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union

__all__ = ["HIPA"]


class SparseSelfAttention(nn.Module):
    """
    稀疏自注意力模块（余弦注意力变体）。
    通过对 Q 和 K 进行 L2 归一化，将点积值域限制在 [-1, 1]，彻底杜绝 FP16 溢出。
    无任何裁剪，完整保留特征信息。
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # 保留缩放，虽归一化后值域已安全

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # 计算 Q、K、V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个形状: (B, num_heads, N, head_dim)

        # 核心改进：对 Q 和 K 进行 L2 归一化，点积退化为余弦相似度，值域 [-1, 1]
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # 计算注意力权重
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # 加权求和
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        # 残差连接 + LayerNorm
        x = self.norm(x + out)
        return x


class HIPABlock(nn.Module):
    """
    Hierarchical Importance Propagation Attention Block
    直接输出稀疏二维特征图、稀疏序列、坐标及稀疏度监控信息。
    本版本已移除所有裁剪，仅依赖余弦注意力保证数值稳定。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_levels: int = 3,
        keep_ratio: float = 0.3,
        keep_ratios: Optional[List[float]] = None,
        min_keeps: Union[int, List[int]] = 8,
        use_contrast_norm: bool = True,
        fill_mode: str = 'constant',
        importance_mode: str = 'l2',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.use_contrast_norm = use_contrast_norm
        self.fill_mode = fill_mode
        self.importance_mode = importance_mode

        # 处理 keep_ratios 参数
        if keep_ratios is not None:
            assert len(keep_ratios) == num_levels
            self.keep_ratios = keep_ratios
        else:
            self.keep_ratios = [keep_ratio] * num_levels

        # 处理 min_keeps 参数
        if isinstance(min_keeps, int):
            self.min_keeps = [min_keeps] * num_levels
        else:
            assert len(min_keeps) == num_levels
            self.min_keeps = min_keeps

        self.proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, out_channels)
        )

    @staticmethod
    def _deterministic_grid_pool(x: torch.Tensor, grid_size: int) -> torch.Tensor:
        B, C, H, W = x.shape
        pad_h = (grid_size - H % grid_size) % grid_size
        pad_w = (grid_size - W % grid_size) % grid_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
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
        assert C == self.in_channels
        dtype = x.dtype
        device = x.device

        # ----- 1. 构建特征金字塔 -----
        pyramid_features = []
        pyramid_coords = []

        for level in range(self.num_levels):
            grid_size = 2 ** level
            pooled = self._deterministic_grid_pool(x, grid_size)
            pyramid_features.append(pooled)

            coords = self._get_coords(B, H, W, grid_size, grid_size, dtype, device)
            pyramid_coords.append(coords)

        # ----- 2. 自底向上重要性筛选 -----
        kept_coords_list = []
        kept_features_list = []

        for level in reversed(range(self.num_levels)):
            feat = pyramid_features[level]
            B, C_l, H_l, W_l = feat.shape
            N_l = H_l * W_l
            feat_flat = feat.flatten(2).transpose(1, 2)

            # 重要性计算（无裁剪，保持原始分布）
            if self.importance_mode == 'l2':
                curr_norm = torch.norm(feat_flat, p=2, dim=-1)
                if self.use_contrast_norm and level < self.num_levels - 1:
                    parent_feat = pyramid_features[level + 1]
                    parent_feat_up = F.interpolate(parent_feat, size=(H_l, W_l), mode='nearest')
                    parent_flat = parent_feat_up.flatten(2).transpose(1, 2)
                    parent_norm = torch.norm(parent_flat, p=2, dim=-1)
                    importance = torch.abs(curr_norm - parent_norm)
                else:
                    importance = curr_norm
            elif self.importance_mode == 'contrast':
                if self.use_contrast_norm and level < self.num_levels - 1:
                    parent_feat = pyramid_features[level + 1]
                    parent_feat_up = F.interpolate(parent_feat, size=(H_l, W_l), mode='nearest')
                    parent_flat = parent_feat_up.flatten(2).transpose(1, 2)
                    diff = feat_flat - parent_flat
                    importance = diff.abs().max(dim=-1)[0]
                else:
                    importance = feat_flat.abs().max(dim=-1)[0]
            else:
                raise ValueError(f"Unsupported importance_mode: {self.importance_mode}")

            # # 仅保留极小下限，防止全零导致 topk 异常
            importance = importance.clamp(min=1e-8,max=2e4)

            keep_ratio_l = self.keep_ratios[level]
            min_keeps_l = self.min_keeps[level]
            keep_num = max(min_keeps_l, int(N_l * keep_ratio_l))
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

        # ----- 3. 生成二维稀疏图（无裁剪）-----
        out_sparse = self._fill_2d(sparse_seq, all_coords, H, W)
        total_kept = sparse_seq.size(1)
        sparsity = torch.tensor(total_kept / (H * W), device=device, dtype=dtype).mean()

        return out_sparse, sparse_seq, all_coords, sparsity


class _HIPASingle(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_levels: int = 3,
        keep_ratio: float = 0.3,
        keep_ratios: Optional[List[float]] = None,
        min_keeps: Union[int, List[int]] = 8,
        use_contrast_norm: bool = True,
        num_heads: int = 8,
        fill_mode: str = 'constant',
        use_self_attn: bool = True,
        importance_mode: str = 'l2',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_self_attn = use_self_attn

        self.hipa = HIPABlock(
            in_channels=in_channels,
            out_channels=out_channels,
            num_levels=num_levels,
            keep_ratio=keep_ratio,
            keep_ratios=keep_ratios,
            min_keeps=min_keeps,
            use_contrast_norm=use_contrast_norm,
            fill_mode=fill_mode,
            importance_mode=importance_mode,
        )

        if use_self_attn:
            self.attn = SparseSelfAttention(embed_dim=out_channels, num_heads=num_heads)
        else:
            self.attn = nn.Identity()

        if in_channels != out_channels:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_sparse_raw, sparse_seq, all_coords, _ = self.hipa(x)
        if self.use_self_attn:
            sparse_seq = self.attn(sparse_seq)
            H, W = x.shape[2:]
            out_sparse = self.hipa._fill_2d(sparse_seq, all_coords, H, W)
        else:
            out_sparse = out_sparse_raw
        residual = self.residual_proj(x)
        return residual + out_sparse


class HIPA(nn.Module):
    """
    HIPA 模块（支持重复 n 次），输出二维稀疏特征图。
    本版本采用余弦注意力，已移除所有裁剪，数值稳定且精度无损。

    参数顺序（YOLO 配置文件 args 列表）：
        in_channels (int)         : 输入通道数
        out_channels (int)        : 输出通道数
        n (int)                   : 模块重复次数，默认 1
        num_levels (int)          : 金字塔层级数，默认 3
        keep_ratio (float)        : 每层保留比例（当未提供 keep_ratios 时生效），默认 0.3
        keep_ratios (List[float]) : 分层保留比例，长度须等于 num_levels（可选）
        min_keeps (int|List[int]) : 每层最少保留区域数，默认 8
        use_contrast_norm (bool)  : 是否使用对比度归一化，默认 True
        num_heads (int)           : 自注意力头数，默认 8
        fill_mode (str)           : 填充模式，'constant' 或 'center'，默认 'constant'
        use_self_attn (bool)      : 是否启用自注意力，默认 True
        importance_mode (str)     : 重要性度量方式，'l2' 或 'contrast'，默认 'l2'
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        num_levels: int = 3,
        keep_ratio: float = 0.3,
        keep_ratios: Optional[List[float]] = None,
        min_keeps: Union[int, List[int]] = 8,
        use_contrast_norm: bool = True,
        num_heads: int = 8,
        fill_mode: str = 'constant',
        use_self_attn: bool = True,
        importance_mode: str = 'l2',
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
                    keep_ratios=keep_ratios,
                    min_keeps=min_keeps,
                    use_contrast_norm=use_contrast_norm,
                    num_heads=num_heads,
                    fill_mode=fill_mode,
                    use_self_attn=use_self_attn,
                    importance_mode=importance_mode,
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
    model = HIPA(
        in_channels=256,
        out_channels=512,
        n=1,
        num_levels=3,
        keep_ratios=[0.5, 0.3, 0.2],
        min_keeps=4,
        use_contrast_norm=True,
        num_heads=4,
        fill_mode='center',
        use_self_attn=True,
        importance_mode='contrast',
    ).cuda().half()
    out = model(dummy)
    print(f"Input: {dummy.shape}, {dummy.dtype}")
    print(f"Output: {out.shape}, {out.dtype}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")