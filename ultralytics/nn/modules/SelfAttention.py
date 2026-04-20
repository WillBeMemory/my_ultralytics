import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    轻量级自注意力模块，用于处理 2D 特征图。
    输入: (B, C, H, W)
    输出: (B, C, H, W)
    """
    def __init__(self, channels, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0., use_ln=True):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert self.head_dim * num_heads == channels, "channels must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

        if use_ln:
            self.norm = nn.LayerNorm(channels)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W  # 序列长度
        # 转换为序列 (B, N, C)
        x_flat = x.flatten(2).transpose(1, 2)  # (B, N, C)

        # 可选 LayerNorm
        x_norm = self.norm(x_flat)

        # 生成 Q, K, V
        qkv = self.qkv(x_norm)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, nh, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 加权求和
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        # 残差连接
        out = out + x_flat

        # 恢复形状
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out


if __name__ == '__main__':
    # 简单测试
    model = SelfAttention(64, num_heads=4)
    x = torch.randn(2, 64, 32, 32)
    out = model(x)
    print(out.shape)  # 应输出 torch.Size([2, 64, 32, 32])

if __name__ == '__main__':
    model = SelfAttention(64, num_heads=4)
    x = torch.randn(2, 64, 32, 32)
    out = model(x)
    print(out.shape)  # torch.Size([2, 64, 32, 32])