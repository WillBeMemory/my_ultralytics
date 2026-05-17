import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Tuple


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
    """
    层级重要性传播注意力模块（增强模式，无门控/填充）。
    多级金字塔重要性筛选 → 保留原始特征（投影）→ 输出序列供自注意力使用。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_levels: int = 3,
        keep_ratio: float = 0.3,
        keep_ratios: Optional[List[float]] = None,
        use_contrast_norm: bool = True,
        importance_mode: str = 'l2',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.use_contrast_norm = use_contrast_norm
        self.importance_mode = importance_mode

        # 分层保留比例
        if keep_ratios is not None and len(keep_ratios) > 0:
            assert len(keep_ratios) == num_levels
            self.keep_ratios = keep_ratios
        else:
            self.keep_ratios = [keep_ratio] * num_levels

        # 投影层（用于自注意力序列的特征变换）
        self.proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, out_channels)
        )

        self._fill_2d_center = self._fill_2d_center

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

    def _fill_2d_center(self, features, coords, H, W):
        """将特征向量散射到对应网格中心（用于自注意力输出）"""
        B, K, C = features.shape
        out = torch.zeros(B, self.out_channels, H, W, device=features.device, dtype=features.dtype)
        cx = (coords[..., 0] * W).long().clamp(0, W - 1)
        cy = (coords[..., 1] * H).long().clamp(0, H - 1)
        src = features.transpose(1, 2)
        index = (cy.unsqueeze(1) * W + cx.unsqueeze(1)).expand(-1, self.out_channels, -1)
        out = out.flatten(2).scatter_(2, index, src).view(B, self.out_channels, H, W)
        return out

    def _hipa_core(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        # 构建金字塔
        pyramid_features, pyramid_coords = [], []
        for level in range(self.num_levels):
            grid_size = 2 ** level
            pooled = self._deterministic_grid_pool(x, grid_size)
            pyramid_features.append(pooled)
            coords = self._get_coords(B, H, W, grid_size, grid_size, dtype, device)
            pyramid_coords.append(coords)

        # 重要性筛选 (自底向上)
        kept_coords_list, kept_features_list = [], []
        for level in reversed(range(self.num_levels)):
            feat = pyramid_features[level]
            B, C_l, H_l, W_l = feat.shape
            N_l = H_l * W_l
            feat_flat = feat.flatten(2).transpose(1, 2)

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

            importance = importance.clamp(min=1e-8, max=2e4)
            keep_ratio_l = self.keep_ratios[level]
            keep_num = max(1, int(N_l * keep_ratio_l))
            keep_num = min(keep_num, N_l)
            topk_indices = torch.topk(importance, k=keep_num, dim=-1)[1]

            kept_feat = torch.gather(feat_flat, 1, topk_indices.unsqueeze(-1).expand(-1, -1, C_l))
            kept_feat = self.proj(kept_feat)
            kept_coords = torch.gather(pyramid_coords[level], 1,
                                       topk_indices.unsqueeze(-1).expand(-1, -1, 4))
            kept_coords_list.insert(0, kept_coords)
            kept_features_list.insert(0, kept_feat)

        sparse_seq = torch.cat(kept_features_list, dim=1)
        all_coords = torch.cat(kept_coords_list, dim=1)
        total_kept = sparse_seq.size(1)
        sparsity = torch.tensor(total_kept / (H * W), device=device, dtype=dtype).mean()

        return None, sparse_seq, all_coords, sparsity

    def forward(self, x: torch.Tensor):
        return self._hipa_core(x)


class _HIPASingle(nn.Module):
    """单层 HIPA 封装：稀疏序列 + 可选位置编码 + 可选自注意力 + 残差连接 + 门控（简化分支用 keep_ratio 计算保留数）"""
    def __init__(
        self, in_channels, out_channels, num_levels=3, keep_ratio=0.3,
        keep_ratios=None, use_contrast_norm=True, num_heads=8,
        use_self_attn=True, importance_mode='l2',
        use_positional_encoding=False,
        use_gate: bool = False,
        gate_reduction: int = 4,
    ):
        super().__init__()
        self.use_self_attn = use_self_attn
        self.use_pos = use_positional_encoding
        self.use_gate = use_gate
        self.out_channels = out_channels
        self.keep_ratio = keep_ratio

        self.hipa = HIPABlock(
            in_channels=in_channels, out_channels=out_channels,
            num_levels=num_levels, keep_ratio=keep_ratio,
            keep_ratios=keep_ratios,
            use_contrast_norm=use_contrast_norm,
            importance_mode=importance_mode,
        )

        if use_self_attn:
            self.attn = SparseSelfAttention(embed_dim=out_channels, num_heads=num_heads)
        else:
            self.attn = nn.Identity()

        self.residual_proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        if self.use_pos:
            self.pos_encoder = nn.Linear(4, out_channels)
        else:
            self.pos_encoder = None

        if use_gate:
            self.gate_net = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, max(1, in_channels // gate_reduction)),
                nn.ReLU(inplace=True),
                nn.Linear(max(1, in_channels // gate_reduction), 1),
                nn.Sigmoid()
            )
        else:
            self.gate_net = None

    def _simple_branch(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype
        importance = torch.norm(x.flatten(2), p=2, dim=1)
        k = max(1, int(H * W * self.keep_ratio))
        k = min(k, H * W)
        topk_val, topk_idx = torch.topk(importance, k, dim=-1)
        x_flat = x.flatten(2).transpose(1, 2)
        features = torch.gather(x_flat, 1, topk_idx.unsqueeze(-1).expand(-1, -1, C))
        features = self.hipa.proj(features)

        ys = (topk_idx.float() / W).floor()
        xs = (topk_idx.float() % W).float()
        cy = (ys + 0.5) / H
        cx = (xs + 0.5) / W
        w = torch.full_like(cx, 1.0 / W)
        h = torch.full_like(cy, 1.0 / H)
        coords = torch.stack([cx, cy, w, h], dim=-1)

        if self.use_pos and self.pos_encoder is not None:
            pos_embed = self.pos_encoder(coords)
            features = features + pos_embed

        if self.use_self_attn and not isinstance(self.attn, nn.Identity):
            features = self.attn(features)

        out = torch.zeros(B, self.out_channels, H, W, device=device, dtype=dtype)
        cx_pix = (coords[..., 0] * W).long().clamp(0, W - 1)
        cy_pix = (coords[..., 1] * H).long().clamp(0, H - 1)
        src = features.transpose(1, 2)
        index = (cy_pix.unsqueeze(1) * W + cx_pix.unsqueeze(1)).expand(-1, self.out_channels, -1)
        out = out.flatten(2).scatter_(2, index, src).view(B, self.out_channels, H, W)
        return out

    def _complex_branch(self, x: torch.Tensor) -> torch.Tensor:
        _, sparse_seq, coords, _ = self.hipa(x)
        if self.use_self_attn and not isinstance(self.attn, nn.Identity) and sparse_seq.size(1) > 0:
            if self.use_pos and self.pos_encoder is not None:
                pos_embed = self.pos_encoder(coords)
                sparse_seq = sparse_seq + pos_embed
            sparse_seq = self.attn(sparse_seq)
        H, W = x.shape[2:]
        out = self.hipa._fill_2d_center(sparse_seq, coords, H, W)
        return out

    def forward(self, x):
        out_complex = self._complex_branch(x)

        if self.use_gate and self.gate_net is not None:
            out_simple = self._simple_branch(x)
            alpha = self.gate_net(x)  # (B, 1)
            alpha = alpha.view(-1, 1, 1, 1)
            out_enhanced = alpha * out_complex + (1 - alpha) * out_simple
        else:
            out_enhanced = out_complex

        return self.residual_proj(x) + out_enhanced


class HIPA(nn.Module):
    """
    可重复多次的 HIPA 模块（YOLO 接口），n 仅占位不重复执行。
    支持可选空间位置编码和门控（简化分支比率由 keep_ratio 控制）
    """
    def __init__(
        self, c1, c2, n=1, num_levels=3, keep_ratio=0.3,
        keep_ratios=None, use_contrast_norm=True, num_heads=8,
        use_self_attn=True, importance_mode='l2',
        use_positional_encoding=True,
        use_gate: bool = True,
        gate_reduction: int = 4,
    ):
        super().__init__()
        # n 占位，不使用
        self.n = n
        self.block = _HIPASingle(
            in_channels=c1, out_channels=c2,
            num_levels=num_levels, keep_ratio=keep_ratio,
            keep_ratios=keep_ratios,
            use_contrast_norm=use_contrast_norm, num_heads=num_heads,
            use_self_attn=use_self_attn,
            importance_mode=importance_mode,
            use_positional_encoding=use_positional_encoding,
            use_gate=use_gate,
            gate_reduction=gate_reduction,
        )

    def forward(self, x):
        return self.block(x)


# ---------- 简单测试 ----------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(2, 256, 20, 20).to(device)

    # 启用门控，n 占位但实际只执行一次
    model_gate = HIPA(256, 512, n=2, keep_ratios=[0.5, 0.3, 0.2],
                      use_self_attn=True, num_heads=8, use_positional_encoding=True,
                      use_gate=True).to(device)
    y1 = model_gate(x)
    print(f"门控版本输出形状: {y1.shape}")

    # 关闭门控（原版行为）
    model_no_gate = HIPA(256, 512, n=1, keep_ratios=[0.5, 0.3, 0.2],
                         use_self_attn=True, num_heads=8, use_positional_encoding=True,
                         use_gate=False).to(device)
    y2 = model_no_gate(x)
    print(f"无门控版本输出形状: {y2.shape}")

    loss = y1.sum() + y2.sum()
    loss.backward()
    print("反向传播通过，测试完毕。")