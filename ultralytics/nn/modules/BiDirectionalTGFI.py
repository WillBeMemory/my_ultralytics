import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BiDirectionalTGFI","BiDirectionalTGFIBlock"]
class BiDirectionalTGFI(nn.Module):
    """
    双向 Top‑K 全局特征交互模块（单块）
    参数:
        dim: 特征通道数（输入输出相同）
        top_k_high: 选取 L2 范数最高的 token 数量
        top_k_low: 选取 L2 范数最低的 token 数量
        num_heads: 多头注意力头数
        attn_drop: 注意力 dropout
        proj_drop: 输出投影 dropout
    """
    def __init__(self, dim, top_k_high=64, top_k_low=64, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.top_k_high = top_k_high
        self.top_k_low = top_k_low
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        # 展平为序列
        x_seq = x.flatten(2).transpose(1, 2)  # (B, N, C)

        # 1. 计算 L2 范数
        l2_norm = torch.norm(x_seq, p=2, dim=-1)  # (B, N)

        # 2. 选择最高和最低索引
        _, idx_high = torch.topk(l2_norm, self.top_k_high, dim=-1)
        _, idx_low = torch.topk(-l2_norm, self.top_k_low, dim=-1)
        all_idx = torch.cat([idx_high, idx_low], dim=1)  # (B, K)

        # 3. 提取对应 token
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, all_idx.size(1))
        x_selected = x_seq[batch_indices, all_idx]  # (B, K, C)

        # 4. 多头自注意力
        qkv = self.qkv(x_selected).reshape(B, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_selected_out = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x_selected_out = self.proj(x_selected_out)
        x_selected_out = self.proj_drop(x_selected_out)

        # 5. 放回原位置
        out_seq = torch.zeros_like(x_seq)
        out_seq.scatter_(1, all_idx.unsqueeze(-1).expand(-1, -1, C), x_selected_out)

        # 6. 恢复形状
        out = out_seq.transpose(1, 2).reshape(B, C, H, W)
        return out


class BiDirectionalTGFIBlock(nn.Module):
    """
    可重复堆叠的 BiDirectionalTGFI 块，支持输入输出通道不同和残差连接。
    参数:
        c1: 输入通道数
        c2: 输出通道数
        n: 内部 TGFI 块的重复次数
        top_k_high: 选取高范数 token 数量
        top_k_low: 选取低范数 token 数量
        num_heads: 注意力头数
        shortcut: 是否使用残差连接（仅当 c1==c2 时有效）
        mid_channels: 中间通道数（若为 None，则取 max(c1,c2)）
    """
    def __init__(self, c1, c2, n=1, top_k_high=64, top_k_low=64, num_heads=8,
                 shortcut=True, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = max(c1, c2)
        self.mid_channels = mid_channels
        self.shortcut = shortcut and (c1 == c2)

        # 输入投影
        self.in_proj = nn.Conv2d(c1, mid_channels, 1) if c1 != mid_channels else nn.Identity()
        # 输出投影
        self.out_proj = nn.Conv2d(mid_channels, c2, 1) if mid_channels != c2 else nn.Identity()

        # 堆叠 n 个 TGFI 块
        self.tgfi_blocks = nn.Sequential(*[
            BiDirectionalTGFI(mid_channels, top_k_high=top_k_high, top_k_low=top_k_low, num_heads=num_heads)
            for _ in range(n)
        ])

    def forward(self, x):
        identity = x
        x = self.in_proj(x)
        x = self.tgfi_blocks(x)
        if self.shortcut:
            x = x + identity
        x = self.out_proj(x)
        return x


# 测试代码
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 测试通道相同，n=2
    model_same = BiDirectionalTGFIBlock(64, 64, n=2, top_k_high=64, top_k_low=64, num_heads=8).to(device)
    x = torch.randn(2, 64, 32, 32).to(device)
    out = model_same(x)
    print(f"Same channel - Input shape: {x.shape}, Output shape: {out.shape}")

    # 测试通道不同 (64 -> 128)
    model_diff = BiDirectionalTGFIBlock(64, 128, n=1, top_k_high=64, top_k_low=64, num_heads=8).to(device)
    out2 = model_diff(x)
    print(f"Diff channel - Input shape: {x.shape}, Output shape: {out2.shape}")

    # 梯度测试
    loss = out.mean() + out2.mean()
    loss.backward()
    print("Backward pass completed.")