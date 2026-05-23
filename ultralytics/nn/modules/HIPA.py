import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union


# ================== 确定性窗口注意力 ==================
class SparseWindowAttention(nn.Module):
    """
    稀疏窗口注意力（确定性实现：整数坐标裁切 + interpolate 缩放）

    在 HIPA 筛选出的高响应坐标周围生成动态窗口（整数像素边界），
    裁切窗口后统一插值到固定尺寸，在窗口内进行多头自注意力，
    最后聚合回中心 token，散射回空间图。
    """

    def __init__(self, channels, num_heads=4, win_size=7, expand_ratio=1.5):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.win_size = win_size
        self.expand_ratio = expand_ratio

        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, dropout=0.0, batch_first=True
        )
        self.pos_embed = nn.Parameter(torch.randn(1, win_size * win_size, channels) * 0.02)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x, coords):
        B, C, H, W = x.shape
        K = coords.shape[1]
        if K == 0:
            return x

        device = x.device
        dtype = x.dtype

        # ---------- 1. 计算整数像素窗口边界 ----------
        cx = coords[..., 0] * W  # (B, K)，中心 x 像素坐标
        cy = coords[..., 1] * H  # 中心 y 像素坐标
        bw = coords[..., 2] * W  # 原始窗口宽度（像素）
        bh = coords[..., 3] * H  # 原始窗口高度（像素）

        half_w = (bw * self.expand_ratio) / 2.0
        half_h = (bh * self.expand_ratio) / 2.0

        x1 = (cx - half_w).floor().long().clamp(0, W - 1)
        x2 = (cx + half_w).ceil().long().clamp(0, W)
        y1 = (cy - half_h).floor().long().clamp(0, H - 1)
        y2 = (cy + half_h).ceil().long().clamp(0, H)

        # ---------- 2. 逐 token 裁切窗口并缩放到固定尺寸 ----------
        window_list = []
        for b in range(B):
            for k in range(K):
                win = x[b, :, y1[b, k]:y2[b, k], x1[b, k]:x2[b, k]]  # (C, h_win, w_win)
                h_win, w_win = win.shape[1], win.shape[2]
                if h_win == 0 or w_win == 0:
                    win = torch.zeros(C, self.win_size, self.win_size, device=device, dtype=dtype)
                else:
                    win = F.interpolate(
                        win.unsqueeze(0),  # (1, C, h_win, w_win)
                        size=(self.win_size, self.win_size),
                        mode='bilinear',
                        align_corners=True  # 关键：确保确定性
                    ).squeeze(0)  # (C, win_size, win_size)
                window_list.append(win)

        window_feat = torch.stack(window_list, dim=0)  # (B*K, C, win_size, win_size)

        # ---------- 3. 展平为序列 (B*K, L, C)，做自注意力 ----------
        window_seq = window_feat.flatten(2).permute(0, 2, 1)  # (B*K, L, C)
        L = self.win_size * self.win_size
        window_seq = window_seq + self.pos_embed[:, :L, :]
        attn_out, _ = self.attn(window_seq, window_seq, window_seq)
        attn_out = self.norm(window_seq + attn_out)

        # ---------- 4. 聚合为中心 token ----------
        center_idx = L // 2
        enhanced_token = attn_out[:, center_idx, :]  # (B*K, C)
        enhanced_token = enhanced_token.reshape(B, K, C)  # (B, K, C)

        # ---------- 5. 散射回空间图 ----------
        out = torch.zeros(B, C, H, W, device=device, dtype=dtype)
        cx_pix = (coords[..., 0] * W).long().clamp(0, W - 1)  # (B, K)
        cy_pix = (coords[..., 1] * H).long().clamp(0, H - 1)  # (B, K)

        index = (cy_pix * W + cx_pix)  # (B, K)
        index = index.unsqueeze(1).expand(B, C, K)  # (B, C, K)
        src = enhanced_token.permute(0, 2, 1)  # (B, C, K)

        out = out.flatten(2).scatter_(2, index, src).view(B, C, H, W)

        # 残差连接
        return out + x


# ================== HIPABlock（保持不变） ==================
class HIPABlock(nn.Module):
    """层级重要性传播注意力模块（无门控）"""

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
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.use_contrast_norm = use_contrast_norm
        self.fill_mode = fill_mode
        self.importance_mode = importance_mode

        if keep_ratios is not None and len(keep_ratios) > 0:
            self.keep_ratios = keep_ratios
        else:
            self.keep_ratios = [keep_ratio] * num_levels

        if isinstance(min_keeps, int):
            self.min_keeps = [min_keeps] * num_levels
        else:
            self.min_keeps = min_keeps

        self.proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, out_channels)
        )

        if fill_mode == 'conv_smooth':
            self.smooth_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False)
        else:
            self.smooth_conv = None

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
        B, K, C = features.shape
        out = torch.zeros(B, self.out_channels, H, W, device=features.device, dtype=features.dtype)
        cx = (coords[..., 0] * W).long().clamp(0, W - 1)
        cy = (coords[..., 1] * H).long().clamp(0, H - 1)
        src = features.transpose(1, 2).to(out.dtype)
        index = (cy.unsqueeze(1) * W + cx.unsqueeze(1)).expand(-1, self.out_channels, -1)
        out = out.flatten(2).scatter_(2, index, src).view(B, self.out_channels, H, W)
        return out

    def _fill_2d_constant(self, features, coords, H, W):
        B, K, C = features.shape
        out = torch.zeros(B, self.out_channels, H, W, device=features.device, dtype=features.dtype)
        for b in range(B):
            for i in range(K):
                cx, cy, w, h = coords[b, i].tolist()
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

    def _fill_2d(self, features, coords, H, W):
        if self.fill_mode == 'center':
            base = self._fill_2d_center(features, coords, H, W)
        elif self.fill_mode == 'constant':
            base = self._fill_2d_constant(features, coords, H, W)
        elif self.fill_mode == 'conv_smooth':
            base = self._fill_2d_center(features, coords, H, W)
        else:
            raise ValueError(f"Unsupported fill_mode: {self.fill_mode}")
        if self.fill_mode == 'conv_smooth' and self.smooth_conv is not None:
            base = self.smooth_conv(base)
        return base

    def _hipa_core(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        pyramid_features, pyramid_coords = [], []
        for level in range(self.num_levels):
            grid_size = 2 ** level
            pooled = self._deterministic_grid_pool(x, grid_size)
            pyramid_features.append(pooled)
            coords = self._get_coords(B, H, W, grid_size, grid_size, dtype, device)
            pyramid_coords.append(coords)

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
        out_sparse = self._fill_2d(sparse_seq, all_coords, H, W)
        return out_sparse, sparse_seq, all_coords

    def forward(self, x: torch.Tensor):
        return self._hipa_core(x)


# ================== _HIPASingle（无门控，使用窗口注意力） ==================
class _HIPASingle(nn.Module):
    """单层 HIPA 封装：稀疏图 + 窗口自注意力 + 残差连接"""

    def __init__(
        self, in_channels, out_channels, num_levels=3, keep_ratio=0.3,
        keep_ratios=None, min_keeps=8, use_contrast_norm=True, num_heads=4,
        fill_mode='center', use_self_attn=True, importance_mode='l2',
        win_size=7, expand_ratio=1.5
    ):
        super().__init__()
        self.use_self_attn = use_self_attn

        self.hipa = HIPABlock(
            in_channels=in_channels, out_channels=out_channels,
            num_levels=num_levels, keep_ratio=keep_ratio,
            keep_ratios=keep_ratios, min_keeps=min_keeps,
            use_contrast_norm=use_contrast_norm, fill_mode=fill_mode,
            importance_mode=importance_mode,
        )

        if use_self_attn:
            self.attn = SparseWindowAttention(
                channels=out_channels,
                num_heads=num_heads,
                win_size=win_size,
                expand_ratio=expand_ratio
            )
        else:
            self.attn = nn.Identity()

        self.residual_proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # 对齐 dtype（AMP 安全）
        target_dtype = self.hipa.proj[1].weight.dtype
        x = x.to(target_dtype)

        out_sparse, sparse_seq, coords = self.hipa(x)

        if self.use_self_attn and not isinstance(self.attn, nn.Identity):
            x_proj = self.residual_proj(x)   # (B, out_channels, H, W)
            out_attn = self.attn(x_proj, coords)  # 窗口注意力
            return x_proj + out_attn          # 残差连接
        else:
            return self.residual_proj(x) + out_sparse


# ================== HIPA 模块（n 占位） ==================
class HIPA(nn.Module):
    def __init__(
        self, c1, c2, n=1, num_levels=3, keep_ratio=0.3,
        keep_ratios=None, min_keeps=8, use_contrast_norm=True, num_heads=4,
        fill_mode='center', use_self_attn=True, importance_mode='l2',
        win_size=7, expand_ratio=1.5
    ):
        super().__init__()
        self.n = n
        blocks = []
        for i in range(n):
            in_ch = c1 if i == 0 else c2
            blocks.append(_HIPASingle(
                in_channels=in_ch, out_channels=c2,
                num_levels=num_levels, keep_ratio=keep_ratio,
                keep_ratios=keep_ratios, min_keeps=min_keeps,
                use_contrast_norm=use_contrast_norm, num_heads=num_heads,
                fill_mode=fill_mode, use_self_attn=use_self_attn,
                importance_mode=importance_mode,
                win_size=win_size, expand_ratio=expand_ratio
            ))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# ================== 简单测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    x = torch.randn(2, 256, 20, 20).to(device)

    # 带窗口注意力
    model_attn = HIPA(256, 512, n=1, keep_ratios=[0.5, 0.3, 0.2], min_keeps=4,
                      use_self_attn=True, num_heads=4, win_size=7, expand_ratio=1.5).to(device)
    y1 = model_attn(x)
    print(f"窗口注意力输出形状: {y1.shape}")

    # 无自注意力
    model_no_attn = HIPA(256, 512, n=1, keep_ratios=[0.5, 0.3, 0.2], min_keeps=4,
                         use_self_attn=False).to(device)
    y2 = model_no_attn(x)
    print(f"无自注意力输出形状: {y2.shape}")

    loss = y1.sum() + y2.sum()
    loss.backward()
    print("反向传播通过，测试完毕。")