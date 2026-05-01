# ============================================
# File: ultralytics/nn/modules/HIPA.py
# 功能：层级重要性传播注意力模块（含自适应背景复杂度）
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
    """

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
    层级重要性传播注意力块。
    新增 adaptive 模式：根据 Haar 高频能量占比判断背景复杂度。
    - 简单背景：使用单尺度 L2 范数选择（保留更多细节，保护小目标）
    - 复杂背景：使用多尺度对比度筛选（抑制建筑纹理）
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
        fill_mode: str = 'center',
        importance_mode: str = 'l2',
        gaussian_sigma_scale: float = 0.3,
        adaptive: bool = False,
        complexity_threshold: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.use_contrast_norm = use_contrast_norm
        self.fill_mode = fill_mode
        self.importance_mode = importance_mode
        self.gaussian_sigma_scale = gaussian_sigma_scale
        self.adaptive = adaptive
        self.complexity_threshold = complexity_threshold

        # 分层保留比例
        if keep_ratios is not None and len(keep_ratios) > 0:
            assert len(keep_ratios) == num_levels
            self.keep_ratios = keep_ratios
        else:
            self.keep_ratios = [keep_ratio] * num_levels

        # 最小保留数
        if isinstance(min_keeps, int):
            self.min_keeps = [min_keeps] * num_levels
        else:
            assert len(min_keeps) == num_levels
            self.min_keeps = min_keeps

        # 投影层（原有多尺度分支使用）
        self.proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, out_channels)
        )

        # 可学习的平滑卷积（conv_smooth 模式）
        if fill_mode == 'conv_smooth':
            self.smooth_conv = nn.Conv2d(
                out_channels, out_channels, kernel_size=3,
                padding=1, groups=out_channels, bias=False
            )
        else:
            self.smooth_conv = None

        # ---------- 自适应相关模块 ----------
        if adaptive:
            # Haar 高频滤波器（用于计算复杂度）
            w_lh = torch.tensor([[1., -1.], [1., -1.]]) / 2.0
            w_hl = torch.tensor([[1., 1.], [-1., -1.]]) / 2.0
            self.register_buffer('haar_lh', w_lh.view(1, 1, 2, 2).repeat(in_channels, 1, 1, 1))
            self.register_buffer('haar_hl', w_hl.view(1, 1, 2, 2).repeat(in_channels, 1, 1, 1))

            # 简化分支所用投影（单尺度）
            self.simple_proj = nn.Sequential(
                nn.LayerNorm(in_channels),
                nn.Linear(in_channels, out_channels)
            )

    # ---------- 金字塔构建辅助方法 ----------
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

    # ---------- 各种填充实现 ----------
    def _fill_2d_center(self, features, coords, H, W):
        B, K, C = features.shape
        out = torch.zeros(B, self.out_channels, H, W, device=features.device, dtype=features.dtype)
        cx = (coords[..., 0] * W).long().clamp(0, W - 1)
        cy = (coords[..., 1] * H).long().clamp(0, H - 1)
        src = features.transpose(1, 2)  # (B, C, K)
        index = (cy.unsqueeze(1) * W + cx.unsqueeze(1)).expand(-1, self.out_channels, -1)
        out = out.flatten(2).scatter_(2, index, src).view(B, self.out_channels, H, W)
        return out

    def _fill_2d_constant(self, features, coords, H, W):
        B, K, C = features.shape
        out = torch.zeros(B, self.out_channels, H, W, device=features.device, dtype=features.dtype)
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

    def _fill_2d_gaussian(self, features, coords, H, W):
        B, K, C = features.shape
        device = features.device
        dtype = features.dtype
        out = torch.zeros(B, self.out_channels, H, W, device=device, dtype=dtype)
        sigma_scale = self.gaussian_sigma_scale
        for b in range(B):
            for k in range(K):
                cx = coords[b, k, 0] * W
                cy = coords[b, k, 1] * H
                w = coords[b, k, 2] * W
                h = coords[b, k, 3] * H
                sigma_x = w * sigma_scale + 1e-6
                sigma_y = h * sigma_scale + 1e-6
                x1 = max(0, int(cx - 3 * sigma_x))
                x2 = min(W, int(cx + 3 * sigma_x + 1))
                y1 = max(0, int(cy - 3 * sigma_y))
                y2 = min(H, int(cy + 3 * sigma_y + 1))
                if x2 <= x1 or y2 <= y1:
                    continue
                y_local = torch.arange(y1, y2, device=device, dtype=dtype) - cy
                x_local = torch.arange(x1, x2, device=device, dtype=dtype) - cx
                with torch.no_grad():
                    gauss = torch.exp(
                        - (x_local.view(1, -1) ** 2) / (2 * sigma_x ** 2)
                        - (y_local.view(-1, 1) ** 2) / (2 * sigma_y ** 2)
                    )
                f = features[b, k]
                out[b, :, y1:y2, x1:x2] += f.view(-1, 1, 1) * gauss
        return out

    def _fill_2d(self, features, coords, H, W):
        if self.fill_mode in ['center', 'conv_smooth']:
            base = self._fill_2d_center(features, coords, H, W)
        elif self.fill_mode == 'constant':
            base = self._fill_2d_constant(features, coords, H, W)
        elif self.fill_mode == 'gaussian':
            base = self._fill_2d_gaussian(features, coords, H, W)
        else:
            raise ValueError(f"Unsupported fill_mode: {self.fill_mode}")
        if self.fill_mode == 'conv_smooth' and self.smooth_conv is not None:
            base = self.smooth_conv(base)
        return base

    def _compute_complexity(self, x):
        """基于 Haar LH/HL 子带能量占比计算背景复杂度"""
        C = self.in_channels
        lh = F.conv2d(x, self.haar_lh, stride=1, padding=0, groups=C)
        hl = F.conv2d(x, self.haar_hl, stride=1, padding=0, groups=C)
        lh = F.pad(lh, (1, 1, 1, 1), mode='reflect')
        hl = F.pad(hl, (1, 1, 1, 1), mode='reflect')
        high_energy = (lh.abs().mean() + hl.abs().mean()) / 2.0
        total_energy = x.abs().mean() + 1e-6
        complexity = high_energy / total_energy
        return complexity

    def _forward_simple(self, x):
        """简化分支：单尺度 L2 范数选择"""
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype
        importance = torch.norm(x, p=2, dim=1).view(B, -1)  # (B, H*W)
        importance = importance.clamp(min=1e-8)
        keep_ratio = self.keep_ratios[0]
        min_keeps = self.min_keeps[0]
        N = H * W
        keep_num = max(min_keeps, int(N * keep_ratio))
        keep_num = min(keep_num, N)
        _, topk_indices = torch.topk(importance, k=keep_num, dim=-1)
        x_flat = x.flatten(2).transpose(1, 2)
        kept_feat = torch.gather(x_flat, 1, topk_indices.unsqueeze(-1).expand(-1, -1, C))
        kept_feat = self.simple_proj(kept_feat)
        # 生成坐标
        y_grid = torch.arange(H, device=device, dtype=dtype).unsqueeze(1).expand(-1, W)
        x_grid = torch.arange(W, device=device, dtype=dtype).unsqueeze(0).expand(H, -1)
        all_coords = torch.stack([x_grid, y_grid], dim=-1).reshape(-1, 2)
        all_coords = all_coords / torch.tensor([W, H], device=device, dtype=dtype)
        coords = all_coords[topk_indices]  # (B, K, 2)
        sizes = torch.tensor([1.0 / W, 1.0 / H], device=device, dtype=dtype).expand(B, keep_num, -1)
        coords = torch.cat([coords, sizes], dim=-1)
        out_sparse = self._fill_2d(kept_feat, coords, H, W)
        sparsity = torch.tensor(keep_num / N, device=device, dtype=dtype).mean()
        return out_sparse, kept_feat, coords, sparsity

    def _standard_forward(self, x):
        """原有前向逻辑（构建金字塔 + 重要性筛选）"""
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        # 1. 构建金字塔
        pyramid_features, pyramid_coords = [], []
        for level in range(self.num_levels):
            grid_size = 2 ** level
            pooled = self._deterministic_grid_pool(x, grid_size)
            pyramid_features.append(pooled)
            coords = self._get_coords(B, H, W, grid_size, grid_size, dtype, device)
            pyramid_coords.append(coords)

        # 2. 重要性筛选 (自底向上)
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
            min_keeps_l = self.min_keeps[level]
            keep_num = max(min_keeps_l, int(N_l * keep_ratio_l))
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

        # 3. 生成稀疏二维图
        out_sparse = self._fill_2d(sparse_seq, all_coords, H, W)
        total_kept = sparse_seq.size(1)
        sparsity = torch.tensor(total_kept / (H * W), device=device, dtype=dtype).mean()
        return out_sparse, sparse_seq, all_coords, sparsity

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        assert C == self.in_channels

        if self.adaptive:
            # 计算复杂度
            complexity = self._compute_complexity(x)
            alpha = torch.sigmoid((self.complexity_threshold - complexity) * 10.0)

            # 复杂背景分支 (原有多尺度)
            out_cplx, seq_cplx, coord_cplx, spar_cplx = self._standard_forward(x)
            # 简单背景分支
            out_simp, seq_simp, coord_simp, spar_simp = self._forward_simple(x)

            # 二维图加权融合
            out_sparse = alpha * out_simp + (1 - alpha) * out_cplx

            # 动态选择序列和坐标
            if alpha > 0.5:
                sparse_seq = seq_simp
                coords = coord_simp
            else:
                sparse_seq = seq_cplx
                coords = coord_cplx

            sparsity = alpha * spar_simp + (1 - alpha) * spar_cplx
            return out_sparse, sparse_seq, coords, sparsity
        else:
            return self._standard_forward(x)


class _HIPASingle(nn.Module):
    """单层 HIPA 封装：稀疏图 + 可选自注意力 + 残差连接"""
    def __init__(
        self, in_channels, out_channels, num_levels=3, keep_ratio=0.3,
        keep_ratios=None, min_keeps=8, use_contrast_norm=True, num_heads=8,
        fill_mode='center', use_self_attn=True, importance_mode='l2',
        gaussian_sigma_scale=0.3, adaptive=False, complexity_threshold=0.3,
    ):
        super().__init__()
        self.use_self_attn = use_self_attn
        self.hipa = HIPABlock(
            in_channels=in_channels, out_channels=out_channels,
            num_levels=num_levels, keep_ratio=keep_ratio,
            keep_ratios=keep_ratios, min_keeps=min_keeps,
            use_contrast_norm=use_contrast_norm, fill_mode=fill_mode,
            importance_mode=importance_mode,
            gaussian_sigma_scale=gaussian_sigma_scale,
            adaptive=adaptive,
            complexity_threshold=complexity_threshold,
        )
        if use_self_attn:
            self.attn = SparseSelfAttention(embed_dim=out_channels, num_heads=num_heads)
        else:
            self.attn = nn.Identity()
        self.residual_proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out_sparse, sparse_seq, coords, _ = self.hipa(x)
        if self.use_self_attn:
            sparse_seq = self.attn(sparse_seq)
            H, W = x.shape[2:]
            out_sparse = self.hipa._fill_2d(sparse_seq, coords, H, W)
        return self.residual_proj(x) + out_sparse


class HIPA(nn.Module):
    """可重复多次的 HIPA 模块（YOLO 接口）"""
    def __init__(
        self, in_channels, out_channels, n=1, num_levels=3, keep_ratio=0.3,
        keep_ratios=None, min_keeps=8, use_contrast_norm=True, num_heads=8,
        fill_mode='center', use_self_attn=True, importance_mode='l2',
        gaussian_sigma_scale=0.3, adaptive=True, complexity_threshold=0.3,
    ):
        super().__init__()
        self.n = n
        blocks = []
        for i in range(n):
            in_ch = in_channels if i == 0 else out_channels
            blocks.append(_HIPASingle(
                in_channels=in_ch, out_channels=out_channels,
                num_levels=num_levels, keep_ratio=keep_ratio,
                keep_ratios=keep_ratios, min_keeps=min_keeps,
                use_contrast_norm=use_contrast_norm, num_heads=num_heads,
                fill_mode=fill_mode, use_self_attn=use_self_attn,
                importance_mode=importance_mode,
                gaussian_sigma_scale=gaussian_sigma_scale,
                adaptive=adaptive,
                complexity_threshold=complexity_threshold,
            ))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# ============================================
# 测试
# ============================================
if __name__ == "__main__":
    for mode in ['center', 'constant', 'conv_smooth']:
        print(f"\n=== Testing fill_mode='{mode}' ===")
        x = torch.randn(2, 256, 20, 20)
        model = HIPA(256, 512, n=1, num_levels=3, keep_ratios=[0.5, 0.3, 0.2],
                     min_keeps=4, fill_mode=mode, use_self_attn=True,
                     importance_mode='contrast', adaptive=True, complexity_threshold=0.3)
        y = model(x)
        print(f"Input: {x.shape} -> Output: {y.shape}")
        print(f"Params: {sum(p.numel() for p in model.parameters()):,}")