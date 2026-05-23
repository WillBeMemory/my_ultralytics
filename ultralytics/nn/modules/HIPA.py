import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv  # YOLO 标准卷积块


# ================== 统一窗口注意力（添加 DWConv 位置编码） ==================
class UniformWindowAttention(nn.Module):
    """
    统一窗口尺寸的稀疏注意力 + 深度可分离位置编码

    以 Top‑K 高响应位置为中心，固定裁剪 win_size×win_size 的窗口，
    在窗口内进行多头自注意力，最终散射回空间图。
    窗口提取后、展平前通过 3×3 DWConv 注入空间位置信息。
    """

    def __init__(self, channels, num_heads=4, keep_ratio=0.3, min_keeps=8,
                 win_size=7, use_pe=True):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.keep_ratio = keep_ratio
        self.min_keeps = min_keeps
        self.win_size = win_size
        self.use_pe = use_pe

        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, dropout=0.0, batch_first=True
        )
        # 简单可学习绝对位置嵌入（保留，与 DWConv PE 互补）
        self.pos_embed = nn.Parameter(torch.randn(1, win_size * win_size, channels) * 0.02)
        self.norm = nn.LayerNorm(channels)

        if self.use_pe:
            # 3×3 深度可分离卷积作为位置编码，零初始化，残差连接
            self.pe_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
            nn.init.zeros_(self.pe_conv.weight)   # 零初始化，让训练逐渐学习
        else:
            self.pe_conv = nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype
        half = self.win_size // 2

        # 1. 反射填充
        x_padded = F.pad(x, (half, half, half, half), mode='reflect')

        # 2. 重要性筛选 Top‑K
        importance = torch.norm(x.flatten(2), p=2, dim=1)          # (B, H*W)
        N = H * W
        k = max(self.min_keeps, int(N * self.keep_ratio))
        k = min(k, N)
        _, topk_indices = torch.topk(importance, k, dim=1)         # (B, K)

        y_coords = (topk_indices // W).long()
        x_coords = (topk_indices % W).long()

        # 3. 批量提取固定窗口
        dy = torch.arange(self.win_size, device=device).view(1, 1, -1, 1)   # (1,1,win_size,1)
        dx = torch.arange(self.win_size, device=device).view(1, 1, 1, -1)   # (1,1,1,win_size)
        y_idx = y_coords.unsqueeze(-1).unsqueeze(-1) + dy
        x_idx = x_coords.unsqueeze(-1).unsqueeze(-1) + dx
        b_idx = torch.arange(B, device=device).view(B, 1, 1, 1)

        # (B, K, C, win_size, win_size)
        windows = x_padded[b_idx, :, y_idx, x_idx]
        windows = windows.permute(0, 1, 4, 2, 3)                     # (B, K, C, win_size, win_size)
        windows = windows.reshape(B * k, C, self.win_size, self.win_size)  # (B*K, C, win_size, win_size)

        # 4. 深度可分离位置编码（残差连接）
        if self.use_pe:
            windows = windows + self.pe_conv(windows)

        # 5. 展平为序列，做自注意力
        L = self.win_size * self.win_size
        window_seq = windows.flatten(2).permute(0, 2, 1)              # (B*K, L, C)
        pos_embed = self.pos_embed[:, :L, :].to(dtype)
        window_seq = window_seq + pos_embed

        attn_out, _ = self.attn(window_seq, window_seq, window_seq)
        attn_out = self.norm(window_seq + attn_out)

        # 6. 聚合为中心 token
        center_idx = L // 2
        enhanced_token = attn_out[:, center_idx, :]                   # (B*K, C)
        enhanced_token = enhanced_token.reshape(B, k, C)              # (B, k, C)

        # 7. 散射回空间图
        out = torch.zeros(B, C, H, W, device=device, dtype=dtype)
        cx_pix = x_coords
        cy_pix = y_coords
        index = (cy_pix * W + cx_pix)                                # (B, k)
        index = index.unsqueeze(1).expand(B, C, k)                   # (B, C, k)
        src = enhanced_token.permute(0, 2, 1).to(dtype)              # (B, C, k)
        out = out.flatten(2).scatter_(2, index, src).view(B, C, H, W)

        return out


# ================== 单层 HIPA（CSP 结构） ==================
class _HIPASingle(nn.Module):
    """
    采用 CSP 结构的 HIPA 单层：
    - cv1 压缩至 2*c（c = int(c2 * e)）
    - 分流 a (恒等) 与 b (窗口注意力增强)
    - cv2 融合输出
    若 c1 == c2，额外添加全局残差连接（可选）。
    """

    def __init__(
        self, c1, c2, e=0.5, num_heads=4, keep_ratio=0.3, min_keeps=8,
        win_size=7, use_pe=True, shortcut=True
    ):
        super().__init__()
        self.c = int(c2 * e)                # CSP 隐藏通道
        self.cv1 = Conv(c1, 2 * self.c, 1)  # 1x1 压缩
        self.cv2 = Conv(2 * self.c, c2, 1)  # 1x1 融合

        # 窗口注意力作用于 b 分支（self.c 通道）
        self.attn = UniformWindowAttention(
            channels=self.c,
            num_heads=num_heads,
            keep_ratio=keep_ratio,
            min_keeps=min_keeps,
            win_size=win_size,
            use_pe=use_pe
        )

        # 是否添加全局残差连接（要求 c1 == c2）
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        # 对齐 dtype（AMP 安全）
        target_dtype = self.cv1.conv.weight.dtype
        x = x.to(target_dtype)

        # 1. CSP 分流
        a, b = self.cv1(x).split((self.c, self.c), dim=1)

        # 2. b 分支经过窗口注意力（保持通道数不变）
        b = b + self.attn(b)                # 残差连接

        # 3. 拼接并融合
        out = self.cv2(torch.cat([a, b], dim=1))

        # 4. 可选全局残差
        if self.shortcut:
            out = out + x
        return out


# ================== HIPA 顶层模块 ==================
class HIPA(nn.Module):
    """
    HIPA 模块（YOLO 接口，支持 CSP + 深度可分离位置编码）

    参数:
        c1, c2 : 输入/输出通道
        n      : 重复次数（占位，默认为 1）
        e      : CSP 压缩比
        num_heads, keep_ratio, min_keeps, win_size : 窗口注意力参数
        use_pe : 是否启用 DWConv 位置编码
        shortcut: 是否添加全局残差连接
    """
    def __init__(
        self, c1, c2, n=1, e=0.5, num_heads=4, keep_ratio=0.5,
        min_keeps=4, win_size=7, use_pe=True, shortcut=True
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
                use_pe=use_pe,
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

    # 模拟 P5 特征图
    x = torch.randn(2, 256, 20, 20).to(device)

    # 实例化 HIPA（CSP + PE）
    model = HIPA(256, 256, n=1, e=0.5, keep_ratio=0.3, min_keeps=4,
                 num_heads=4, win_size=7, use_pe=True, shortcut=True).to(device)
    y = model(x)
    print(f"Input: {x.shape}  Output: {y.shape}")

    loss = y.mean()
    loss.backward()
    print("Backward passed. Test OK!")