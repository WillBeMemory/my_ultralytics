import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv


class SparseGlobalAttention(nn.Module):
    """
    稀疏区域全局注意力

    1. 基于 L2 范数选择 Top‑K 高响应位置。
    2. 以每个位置为中心，固定裁剪 win_size×win_size 的窗口。
    3. 收集所有窗口内的 token，附加位置编码（局部 + 窗口级）。
    4. 在所有 token 组成的序列上执行一次全局多头自注意力。
    5. 将增强后的中心 token 散射回空间图。
    """

    def __init__(self, channels, num_heads=4, keep_ratio=0.3, min_keeps=8, win_size=7):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.keep_ratio = keep_ratio
        self.min_keeps = min_keeps
        self.win_size = win_size

        # 全局多头自注意力
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, dropout=0.0, batch_first=True
        )

        # 局部位置编码（窗口内相对位置）
        self.local_pos = nn.Parameter(torch.randn(1, win_size * win_size, channels) * 0.02)

        # 窗口级位置编码（由中心坐标生成）
        self.win_pos_encoder = nn.Sequential(
            nn.Linear(2, channels),
            nn.SiLU(inplace=True),
            nn.Linear(channels, channels),
        )

        # 输出 LayerNorm
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype
        half = self.win_size // 2

        # ---------- 1. 反射填充，确保窗口不越界 ----------
        x_padded = F.pad(x, (half, half, half, half), mode='reflect')

        # ---------- 2. 重要性筛选 Top‑K ----------
        importance = torch.norm(x.flatten(2), p=2, dim=1)          # (B, H*W)
        N = H * W
        k = max(self.min_keeps, int(N * self.keep_ratio))
        k = min(k, N)
        _, topk_indices = torch.topk(importance, k, dim=1)         # (B, K)

        y_coords = (topk_indices // W).long()
        x_coords = (topk_indices % W).long()

        # ---------- 3. 批量提取固定窗口 ----------
        dy = torch.arange(self.win_size, device=device).view(1, 1, -1, 1)   # (1,1,win_size,1)
        dx = torch.arange(self.win_size, device=device).view(1, 1, 1, -1)   # (1,1,1,win_size)
        y_idx = y_coords.unsqueeze(-1).unsqueeze(-1) + dy
        x_idx = x_coords.unsqueeze(-1).unsqueeze(-1) + dx
        b_idx = torch.arange(B, device=device).view(B, 1, 1, 1)

        # 提取窗口 (B, K, C, win_size, win_size)
        windows = x_padded[b_idx, :, y_idx, x_idx]
        windows = windows.permute(0, 1, 4, 2, 3)                     # (B, K, C, win_size, win_size)
        windows = windows.reshape(B * k, C, self.win_size, self.win_size)  # (B*K, C, win_size, win_size)

        # ---------- 4. 展平 token，附加位置编码 ----------
        L = self.win_size * self.win_size
        tokens = windows.flatten(2).permute(0, 2, 1)                 # (B*K, L, C)

        # 局部位置编码
        local_pos = self.local_pos[:, :L, :].to(dtype)
        tokens = tokens + local_pos

        # 窗口级位置编码
        cx = x_coords.float() / W   # 归一化到 [0,1]
        cy = y_coords.float() / H
        win_coords = torch.stack([cx, cy], dim=-1)                   # (B, K, 2)
        win_pos = self.win_pos_encoder(win_coords.to(dtype))         # (B, K, C)
        # 扩展并加到每个窗口的所有 token 上
        tokens = tokens.reshape(B, k, L, C)
        tokens = tokens + win_pos.unsqueeze(2)                       # (B, K, L, C)
        tokens = tokens.flatten(1, 2)                                 # (B, K*L, C)

        # ---------- 5. 全局自注意力 ----------
        attn_out, _ = self.attn(tokens, tokens, tokens)              # (B, K*L, C)
        attn_out = self.norm(tokens + attn_out)                      # 残差 + 归一化

        # ---------- 6. 聚合为中心 token ----------
        attn_out = attn_out.reshape(B, k, L, C)                      # (B, K, L, C)
        center_idx = L // 2
        enhanced_token = attn_out[:, :, center_idx, :]               # (B, K, C)

        # ---------- 7. 散射回空间图 ----------
        out = torch.zeros(B, C, H, W, device=device, dtype=dtype)
        cx_pix = x_coords
        cy_pix = y_coords
        index = (cy_pix * W + cx_pix)                                # (B, K)
        index = index.unsqueeze(1).expand(B, C, k)                   # (B, C, k)
        src = enhanced_token.permute(0, 2, 1).to(dtype)              # (B, C, k)
        out = out.flatten(2).scatter_(2, index, src).view(B, C, H, W)

        return out


class _HIPASingle(nn.Module):
    """采用 CSP 结构的单层 HIPA（稀疏区域全局注意力版）"""

    def __init__(
        self, c1, c2, e=0.5, num_heads=4, keep_ratio=0.3, min_keeps=8,
        win_size=7, shortcut=True
    ):
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
        b = b + self.attn(b)                     # 残差连接
        out = self.cv2(torch.cat([a, b], dim=1))

        if self.shortcut:
            out = out + x
        return out


class HIPA(nn.Module):
    """HIPA 模块（YOLO 接口，CSP + 稀疏区域全局注意力）"""

    def __init__(
        self, c1, c2, n=1, e=0.5, num_heads=4, keep_ratio=0.3,
        min_keeps=8, win_size=7, shortcut=True
    ):
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

    x = torch.randn(2, 256, 20, 20).to(device)

    model = HIPA(256, 256, n=1, e=0.5, keep_ratio=0.3, min_keeps=4,
                 num_heads=4, win_size=7, shortcut=True).to(device)
    y = model(x)
    print(f"Input: {x.shape}  Output: {y.shape}")

    loss = y.mean()
    loss.backward()
    print("Backward passed. Test OK!")