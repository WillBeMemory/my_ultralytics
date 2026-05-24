import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv


class SparseGlobalAttention(nn.Module):
    def __init__(self, channels, num_heads=4, keep_ratio=0.3, min_keeps=8, win_size=7):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.keep_ratio = keep_ratio
        self.min_keeps = min_keeps
        self.win_size = win_size

        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, dropout=0.0, batch_first=True
        )
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, channels),
            nn.SiLU(inplace=True),
            nn.Linear(channels, channels),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, f):
        B, C, H, W = f.shape
        device = f.device
        half = self.win_size // 2

        # ---- 关键：统一为 pos_encoder 权重的 dtype，保证 matmul 类型一致 ----
        param_dtype = self.pos_encoder[0].weight.dtype
        input_dtype = f.dtype
        f = f.to(param_dtype)

        # 1. 重要性筛选 Top‑K
        importance = torch.norm(f.flatten(2), p=2, dim=1)   # (B, H*W)
        N = H * W
        k = max(self.min_keeps, int(N * self.keep_ratio))
        k = min(k, N)
        _, topk_indices = torch.topk(importance, k, dim=1)

        y_c = (topk_indices // W).long()
        x_c = (topk_indices % W).long()

        # 2. 窗口内绝对坐标
        dy, dx = torch.meshgrid(
            torch.arange(-half, half + 1, device=device, dtype=torch.long),
            torch.arange(-half, half + 1, device=device, dtype=torch.long),
            indexing='ij'
        )
        dy = dy.unsqueeze(0).unsqueeze(0)
        dx = dx.unsqueeze(0).unsqueeze(0)

        y_abs = (y_c.unsqueeze(2).unsqueeze(3) + dy).clamp(0, H - 1)
        x_abs = (x_c.unsqueeze(2).unsqueeze(3) + dx).clamp(0, W - 1)

        y_flat = y_abs.flatten(1, 3)
        x_flat = x_abs.flatten(1, 3)

        b_idx = torch.arange(B, device=device).unsqueeze(1).unsqueeze(1).expand(-1, k, self.win_size * self.win_size)
        b_flat = b_idx.flatten(1, 2)      # 修复维度索引

        coords = torch.stack([b_flat, y_flat, x_flat], dim=-1).reshape(-1, 3)
        unique_coords = torch.unique(coords, dim=0)

        uni_b = unique_coords[:, 0].long()
        uni_y = unique_coords[:, 1].long()
        uni_x = unique_coords[:, 2].long()

        num_unique = unique_coords.shape[0]
        if num_unique == 0:
            return torch.zeros_like(f).to(input_dtype)

        # 3. 提取特征并附加位置编码
        feats = f[uni_b, :, uni_y, uni_x]                # (num_unique, C)，dtype = param_dtype

        x_norm = uni_x.float() / max(W - 1, 1)
        y_norm = uni_y.float() / max(H - 1, 1)
        coords_norm = torch.stack([x_norm, y_norm], dim=-1).to(param_dtype)  # 保证与 pos_encoder 一致
        pos_embed = self.pos_encoder(coords_norm)        # dtype = param_dtype
        feats = feats + pos_embed

        # 4. 逐 batch 全局自注意力
        enhanced_feats = feats.clone()
        for b in range(B):
            mask_b = (uni_b == b)
            nb = mask_b.sum().item()
            if nb == 0:
                continue
            seq = feats[mask_b].unsqueeze(0)       # (1, nb, C)
            attn_out, _ = self.attn(seq, seq, seq)
            attn_out = self.norm(seq + attn_out)
            enhanced_feats[mask_b] = attn_out.squeeze(0)

        # 5. 散射回空间图
        out = torch.zeros(B, C, H, W, device=device, dtype=param_dtype)
        out[uni_b, :, uni_y, uni_x] = enhanced_feats

        return out.to(input_dtype)   # 还原为模块输入时的原始 dtype


class _HIPASingle(nn.Module):
    def __init__(self, c1, c2, e=0.5, num_heads=4, keep_ratio=0.3, min_keeps=8,
                 win_size=7, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)

        self.attn = SparseGlobalAttention(
            channels=self.c,
            num_heads=num_heads,
            keep_ratio=keep_ratio,
            min_keeps=min_keeps,
            win_size=win_size
        )

        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        target_dtype = self.cv1.conv.weight.dtype
        x = x.to(target_dtype)

        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)   # 残差连接
        out = self.cv2(torch.cat([a, b], dim=1))

        if self.shortcut:
            out = out + x
        return out


class HIPA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5, num_heads=4, keep_ratio=0.3,
                 min_keeps=8, win_size=7, shortcut=True):
        super().__init__()
        self.n = n
        blocks = []
        for i in range(n):
            in_ch = c1 if i == 0 else c2
            blocks.append(_HIPASingle(
                c1=in_ch, c2=c2, e=e,
                num_heads=num_heads,
                keep_ratio=keep_ratio,
                min_keeps=min_keeps,
                win_size=win_size,
                shortcut=shortcut
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

    feat_map = torch.randn(2, 256, 20, 20).to(device)

    model = HIPA(256, 256, n=1, e=0.5, keep_ratio=0.3, min_keeps=4,
                 num_heads=4, win_size=7, shortcut=True).to(device)
    y = model(feat_map)
    print(f"Input: {feat_map.shape}  Output: {y.shape}")

    loss = y.mean()
    loss.backward()
    print("Backward passed. Test OK!")