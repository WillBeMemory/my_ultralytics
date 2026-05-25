import torch
import torch.nn as nn
import torch.nn.functional as F


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
    高效 HIPA 核心：局部极差选点 + 固定窗口扩张（向量化）
    1. 计算局部极差，top‑k 选取高响应像素。
    2. 以每个选中点为中心生成 win_size × win_size 窗口，收集所有窗口内像素坐标并去重。
    3. 提取去重像素特征，投影，输出 token 序列、坐标和初始散射图。
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        keep_ratio=0.3,
        min_keeps=8,
        local_range_kernel=3,
        win_size=7,               # 窗口扩张大小
        fill_mode='center',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.keep_ratio = keep_ratio
        self.min_keeps = min_keeps
        self.local_range_kernel = local_range_kernel
        self.win_size = win_size
        self.fill_mode = fill_mode

        self.proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, out_channels)
        )
        if fill_mode == 'conv_smooth':
            self.smooth_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False)
        else:
            self.smooth_conv = None

    def compute_local_range(self, feat):
        k = self.local_range_kernel
        max_val = F.max_pool2d(feat, k, stride=1, padding=k // 2)
        min_val = -F.max_pool2d(-feat, k, stride=1, padding=k // 2)
        return (max_val - min_val).mean(dim=1)   # (B, H, W)

    def _fill_2d_center(self, features, coords, H, W):
        """将特征放置在网格中心点（coords 已归一化）"""
        B, K, C = features.shape
        out = torch.zeros(B, self.out_channels, H, W, device=features.device, dtype=features.dtype)
        cx = (coords[..., 0] * W).long().clamp(0, W - 1)
        cy = (coords[..., 1] * H).long().clamp(0, H - 1)
        src = features.transpose(1, 2)
        index = (cy.unsqueeze(1) * W + cx.unsqueeze(1)).expand(-1, self.out_channels, -1)
        out = out.flatten(2).scatter_(2, index, src).view(B, self.out_channels, H, W)
        return out

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        # 1. 局部极差 + top‑k 选点
        range_map = self.compute_local_range(x)      # (B, H, W)
        N = H * W
        flat_range = range_map.flatten(1)            # (B, N)
        k = max(self.min_keeps, int(N * self.keep_ratio))
        k = min(k, N)
        _, topk_idx = torch.topk(flat_range, k=k, dim=-1)   # (B, k)

        # 2. 向量化窗口扩张并去重 (逐样本 GPU 操作)
        half = self.win_size // 2
        # 创建相对偏移
        dy, dx = torch.meshgrid(
            torch.arange(-half, half + 1, device=device, dtype=torch.long),
            torch.arange(-half, half + 1, device=device, dtype=torch.long),
            indexing='ij'
        )  # (win_size, win_size)
        dy = dy.reshape(1, 1, -1)  # (1, 1, win^2)
        dx = dx.reshape(1, 1, -1)

        token_list = []
        coord_list = []
        out_sparse = torch.zeros(B, self.out_channels, H, W, device=device, dtype=x.dtype)

        for b in range(B):
            idx = topk_idx[b]           # (k,)
            yc = idx // W               # (k,)
            xc = idx % W                # (k,)

            # 绝对坐标 (k, win^2)
            y_abs = yc.unsqueeze(1) + dy   # (k, win^2)
            x_abs = xc.unsqueeze(1) + dx
            y_abs = y_abs.clamp(0, H - 1)
            x_abs = x_abs.clamp(0, W - 1)

            # 展平并拼接为 (k*win^2, 2)
            y_flat = y_abs.reshape(-1)
            x_flat = x_abs.reshape(-1)
            coords_flat = torch.stack([y_flat, x_flat], dim=-1)   # (M, 2)

            # 去重
            coords_unique = torch.unique(coords_flat, dim=0)      # (M_unique, 2)
            M = coords_unique.shape[0]

            if M == 0:
                token_list.append(torch.empty(0, self.out_channels, device=device))
                coord_list.append(torch.empty(0, 2, dtype=torch.long, device=device))
                continue

            # 提取特征 (M, C)
            feats = x[b, :, coords_unique[:, 0], coords_unique[:, 1]].transpose(0, 1)  # (M, C)
            feats_proj = self.proj(feats)   # (M, out_channels)
            token_list.append(feats_proj)
            coord_list.append(coords_unique)

            # 初始散射图（用于残差）
            cx = (coords_unique[:, 1].float() + 0.5) / W
            cy = (coords_unique[:, 0].float() + 0.5) / H
            w = torch.full((M,), 1.0 / W, device=device)
            h = torch.full((M,), 1.0 / H, device=device)
            norm_coords = torch.stack([cx, cy, w, h], dim=-1).unsqueeze(0)  # (1, M, 4)
            out_sparse[b] = self._fill_2d_center(feats_proj.unsqueeze(0), norm_coords, H, W)

        sparsity = torch.tensor(sum(t.shape[0] for t in token_list) / (B * N), device=device)
        return out_sparse, token_list, coord_list, sparsity


class _HIPASingle(nn.Module):
    """单层 HIPA：稀疏图 + 逐样本自注意力 + 残差"""
    def __init__(self, in_channels, out_channels,
                 keep_ratio=0.3, min_keeps=8, num_heads=8,
                 win_size=7, fill_mode='center', use_self_attn=True,
                 local_range_kernel=3):
        super().__init__()
        self.use_self_attn = use_self_attn
        self.hipa = HIPABlock(
            in_channels=in_channels, out_channels=out_channels,
            keep_ratio=keep_ratio, min_keeps=min_keeps,
            local_range_kernel=local_range_kernel,
            win_size=win_size, fill_mode=fill_mode
        )
        self.attn = SparseSelfAttention(out_channels, num_heads) if use_self_attn else nn.Identity()
        self.residual_proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        target_dtype = self.hipa.proj[1].weight.dtype
        x = x.to(target_dtype)

        out_sparse, token_list, coord_list, _ = self.hipa(x)

        if self.use_self_attn:
            B, _, H, W = x.shape
            new_out = torch.zeros_like(out_sparse)
            for b in range(B):
                tokens = token_list[b]          # (M, C)
                if tokens.shape[0] == 0:
                    continue
                seq = tokens.unsqueeze(0)       # (1, M, C)
                enhanced = self.attn(seq)       # (1, M, C)
                coords_t = coord_list[b]        # (M, 2)
                cx = (coords_t[:, 1].float() + 0.5) / W
                cy = (coords_t[:, 0].float() + 0.5) / H
                w = torch.full((coords_t.shape[0],), 1.0 / W, device=x.device)
                h = torch.full((coords_t.shape[0],), 1.0 / H, device=x.device)
                norm_coords = torch.stack([cx, cy, w, h], dim=-1).unsqueeze(0)
                new_out[b] = self.hipa._fill_2d_center(enhanced, norm_coords, H, W)
            out_sparse = new_out

        return self.residual_proj(x) + out_sparse


class HIPA(nn.Module):
    """YOLO 接口的 HIPA 模块"""
    def __init__(self, c1, c2, n=1, keep_ratio=0.3, min_keeps=8,
                 num_heads=8, win_size=7, fill_mode='center',
                 use_self_attn=True, local_range_kernel=3):
        super().__init__()
        self.n = n
        blocks = []
        for i in range(n):
            in_ch = c1 if i == 0 else c2
            blocks.append(_HIPASingle(
                in_channels=in_ch, out_channels=c2,
                keep_ratio=keep_ratio, min_keeps=min_keeps,
                num_heads=num_heads, win_size=win_size,
                fill_mode=fill_mode, use_self_attn=use_self_attn,
                local_range_kernel=local_range_kernel
            ))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# 测试
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(2, 256, 20, 20).to(device)
    model = HIPA(256, 512, n=1, keep_ratio=0.3, min_keeps=4,
                 win_size=7, use_self_attn=True).to(device)
    y = model(x)
    print(f"Output shape: {y.shape}")
    loss = y.sum()
    loss.backward()
    print("Test passed.")