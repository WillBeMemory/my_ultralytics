import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union


class SparseSelfAttention(nn.Module):
    """稀疏自注意力（余弦注意力变体）"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
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
    基于局部极差的 HIPA 模块（无金字塔）
    1. 计算局部极差 (max_pool - min_pool)
    2. 选取 top‑k 高极差位置
    3. 投影这些位置的特征到 out_channels
    4. 将保留特征序列与坐标返回，供后续自注意力使用
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        keep_ratio=0.3,
        min_keeps=8,
        local_range_kernel=3,   # 局部极差的窗口大小
        fill_mode='center',     # 散射方式
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.keep_ratio = keep_ratio
        self.min_keeps = min_keeps
        self.local_range_kernel = local_range_kernel
        self.fill_mode = fill_mode

        # 投影层：将保留的特征投影到输出维度
        self.proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, out_channels)
        )

        # 平滑卷积（仅 conv_smooth 模式）
        if fill_mode == 'conv_smooth':
            self.smooth_conv = nn.Conv2d(out_channels, out_channels, 3,
                                         padding=1, groups=out_channels, bias=False)
        else:
            self.smooth_conv = None

    def compute_local_range(self, feat):
        """
        计算局部极差：max_pool - min_pool，沿通道平均
        feat: (B, C, H, W) -> (B, H, W)
        """
        k = self.local_range_kernel
        max_val = F.max_pool2d(feat, k, stride=1, padding=k // 2)
        # 最小值池化：-max_pool(-x)
        min_val = -F.max_pool2d(-feat, k, stride=1, padding=k // 2)
        range_val = (max_val - min_val).mean(dim=1)   # (B, H, W)
        return range_val

    def _fill_2d_center(self, features, coords, H, W):
        B, K, C = features.shape
        out = torch.zeros(B, self.out_channels, H, W, device=features.device, dtype=features.dtype)
        cx = (coords[..., 0] * W).long().clamp(0, W - 1)
        cy = (coords[..., 1] * H).long().clamp(0, H - 1)
        src = features.transpose(1, 2)
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

    def forward(self, x):
        """
        返回：
            out_sparse : (B, out_channels, H, W) 散射回的空间图
            sparse_seq : (B, K, out_channels) 保留的特征序列
            coords : (B, K, 4) 归一化坐标 [cx, cy, w, h]
            sparsity : 稀疏度 (标量)
        """
        B, C, H, W = x.shape
        device = x.device

        # 1. 计算局部极差
        range_map = self.compute_local_range(x)          # (B, H, W)

        # 2. top‑k 选择
        N = H * W
        flat_range = range_map.flatten(1)                # (B, N)
        k = max(self.min_keeps, int(N * self.keep_ratio))
        k = min(k, N)
        topk_idx = torch.topk(flat_range, k=k, dim=-1)[1]  # (B, k)

        # 3. 索引转换为网格坐标 (归一化)
        y_idx = topk_idx // W
        x_idx = topk_idx % W
        grid_h = 1.0 / H
        grid_w = 1.0 / W
        cx = (x_idx.float() + 0.5) * grid_w    # (B, k)
        cy = (y_idx.float() + 0.5) * grid_h
        w = torch.full_like(cx, grid_w)
        h = torch.full_like(cy, grid_h)
        coords = torch.stack([cx, cy, w, h], dim=-1)   # (B, k, 4)

        # 4. 提取对应位置的特征
        feat_flat = x.flatten(2).transpose(1, 2)       # (B, N, C)
        kept_feat = torch.gather(feat_flat, 1,
                                 topk_idx.unsqueeze(-1).expand(B, -1, C))
        kept_feat = self.proj(kept_feat)               # (B, k, out_channels)

        # 5. 散射回空间图（用于残差）
        out_sparse = self._fill_2d(kept_feat, coords, H, W)

        sparsity = torch.tensor(k / N, device=device)
        return out_sparse, kept_feat, coords, sparsity


class _HIPASingle(nn.Module):
    """单层 HIPA 封装：稀疏图 + 自注意力（仅对保留区域） + 残差连接"""
    def __init__(self, in_channels, out_channels,
                 keep_ratio=0.3, min_keeps=8,
                 num_heads=8, fill_mode='center',
                 use_self_attn=True,
                 local_range_kernel=3):
        super().__init__()
        self.use_self_attn = use_self_attn

        self.hipa = HIPABlock(
            in_channels=in_channels,
            out_channels=out_channels,
            keep_ratio=keep_ratio,
            min_keeps=min_keeps,
            local_range_kernel=local_range_kernel,
            fill_mode=fill_mode
        )

        if use_self_attn:
            self.attn = SparseSelfAttention(embed_dim=out_channels, num_heads=num_heads)
        else:
            self.attn = nn.Identity()

        self.residual_proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        target_dtype = self.hipa.proj[1].weight.dtype
        x = x.to(target_dtype)

        out_sparse, sparse_seq, coords, _ = self.hipa(x)

        if self.use_self_attn and sparse_seq is not None:
            sparse_seq = self.attn(sparse_seq)
            H, W = x.shape[2:]
            out_sparse = self.hipa._fill_2d(sparse_seq, coords, H, W)

        return self.residual_proj(x) + out_sparse


class HIPA(nn.Module):
    """可堆叠多次的 HIPA 模块（YOLO 接口）"""
    def __init__(self, c1, c2, n=1, keep_ratio=0.3, min_keeps=8,
                 num_heads=8, fill_mode='center',
                 use_self_attn=True, local_range_kernel=3):
        super().__init__()
        self.n = n
        blocks = []
        for i in range(n):
            in_ch = c1 if i == 0 else c2
            blocks.append(_HIPASingle(
                in_channels=in_ch, out_channels=c2,
                keep_ratio=keep_ratio,
                min_keeps=min_keeps,
                num_heads=num_heads,
                fill_mode=fill_mode,
                use_self_attn=use_self_attn,
                local_range_kernel=local_range_kernel
            ))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# ---------- 测试 ----------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(2, 256, 20, 20).to(device)

    model = HIPA(256, 512, n=1, keep_ratio=0.3, min_keeps=4,
                 num_heads=8, use_self_attn=True, local_range_kernel=3).to(device)
    y = model(x)
    print(f"Output shape: {y.shape}")  # expected (2, 512, 20, 20)

    loss = y.sum()
    loss.backward()
    print("Test passed.")