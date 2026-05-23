import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union


class UniformWindowAttention(nn.Module):
    """
    统一窗口尺寸的稀疏注意力（确定性、训练极快）

    以 Top-K 高响应位置为中心，固定裁剪 win_size×win_size 的窗口，
    在窗口内进行多头自注意力，最后散射回空间图。
    """

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
        self.pos_embed = nn.Parameter(torch.randn(1, win_size * win_size, channels) * 0.02)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype
        half = self.win_size // 2

        # ---------- 1. 反射填充 ----------
        x_padded = F.pad(x, (half, half, half, half), mode='reflect')

        # ---------- 2. 重要性筛选 Top‑K ----------
        importance = torch.norm(x.flatten(2), p=2, dim=1)   # (B, H*W)
        N = H * W
        k = max(self.min_keeps, int(N * self.keep_ratio))
        k = min(k, N)
        _, topk_indices = torch.topk(importance, k, dim=1)  # (B, K)

        y_coords = (topk_indices // W).long()
        x_coords = (topk_indices % W).long()

        # ---------- 3. 构建广播一致的索引 ----------
        # 用于窗口内坐标偏移
        dy = torch.arange(self.win_size, device=device).view(1, 1, -1, 1)   # (1,1,win_size,1)
        dx = torch.arange(self.win_size, device=device).view(1, 1, 1, -1)   # (1,1,1,win_size)

        # 扩展坐标
        y_idx = y_coords.unsqueeze(-1).unsqueeze(-1) + dy    # (B, K, win_size, 1)
        x_idx = x_coords.unsqueeze(-1).unsqueeze(-1) + dx    # (B, K, 1, win_size)

        # 批量索引
        b_idx = torch.arange(B, device=device).view(B, 1, 1, 1)   # (B,1,1,1)

        # 从填充图中提取窗口：形状 (B, K, win_size, win_size, C)
        windows = x_padded[b_idx, :, y_idx, x_idx]
        # 调整维度： (B, K, C, win_size, win_size)
        windows = windows.permute(0, 1, 4, 2, 3)
        # 展平 batch 和 token 维度 -> (B*K, C, win_size, win_size)
        windows = windows.reshape(B * k, C, self.win_size, self.win_size)

        # ---------- 4. 窗口内自注意力 ----------
        L = self.win_size * self.win_size
        window_seq = windows.flatten(2).permute(0, 2, 1)       # (B*K, L, C)
        pos_embed = self.pos_embed[:, :L, :].to(dtype)
        window_seq = window_seq + pos_embed

        # 强制 float32 避免 AMP 内部类型提升
        window_seq_fp32 = window_seq.float()
        attn_out, _ = self.attn(window_seq_fp32, window_seq_fp32, window_seq_fp32)
        attn_out = self.norm(window_seq_fp32 + attn_out)
        attn_out = attn_out.to(dtype)

        # ---------- 5. 聚合中心 token ----------
        center_idx = L // 2
        enhanced_token = attn_out[:, center_idx, :]            # (B*K, C)
        enhanced_token = enhanced_token.reshape(B, k, C)       # (B, k, C)

        # ---------- 6. 散射回空间图 ----------
        out = torch.zeros(B, C, H, W, device=device, dtype=dtype)
        cx_pix = x_coords
        cy_pix = y_coords
        index = (cy_pix * W + cx_pix)                          # (B, k)
        index = index.unsqueeze(1).expand(B, C, k)            # (B, C, k)
        src = enhanced_token.permute(0, 2, 1).to(dtype)       # (B, C, k)
        out = out.flatten(2).scatter_(2, index, src).view(B, C, H, W)

        return out


class _HIPASingle(nn.Module):
    """单层简化 HIPA：Top‑K 窗口注意力 + 残差连接"""

    def __init__(self, in_channels, out_channels,
                 num_heads=4, keep_ratio=0.3, min_keeps=8, win_size=7):
        super().__init__()
        self.attn = UniformWindowAttention(
            channels=in_channels,
            num_heads=num_heads,
            keep_ratio=keep_ratio,
            min_keeps=min_keeps,
            win_size=win_size
        )
        self.proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        target_dtype = self.proj.weight.dtype if isinstance(self.proj, nn.Conv2d) else x.dtype
        x = x.to(target_dtype)
        out_attn = self.attn(x)
        return self.proj(x + out_attn)


class HIPA(nn.Module):
    """HIPA 模块（YOLO 接口，n 占位）"""

    def __init__(self, c1, c2, n=1,
                 num_heads=4, keep_ratio=0.25, min_keeps=4, win_size=7):
        super().__init__()
        self.n = n
        blocks = []
        for i in range(n):
            in_ch = c1 if i == 0 else c2
            blocks.append(_HIPASingle(
                in_channels=in_ch, out_channels=c2,
                num_heads=num_heads,
                keep_ratio=keep_ratio,
                min_keeps=min_keeps,
                win_size=win_size
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

    model = HIPA(256, 512, n=1, keep_ratio=0.3, min_keeps=4, num_heads=4, win_size=7).to(device)
    y = model(x)
    print(f"Input: {x.shape}  Output: {y.shape}")

    loss = y.mean()
    loss.backward()
    print("Test passed!")