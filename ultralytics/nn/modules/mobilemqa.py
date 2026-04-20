import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv  # 确保Conv已正确定义

_all_ = ['A2C2f_MobileMQA']
# ---------- MobileMQA 核心模块（保持不变） ----------
class MobileMQA(nn.Module):
    """Mobile Multi-Query Attention (Mobile MQA) module from MobileNetV4."""
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., sr_ratio=2):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr is not None:
            x_2d = x.transpose(1, 2).reshape(B, C, H, W)
            x_2d = self.sr(x_2d)
            _, _, h_r, w_r = x_2d.shape
            x_kv = x_2d.reshape(B, C, -1).transpose(1, 2)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x
            h_r, w_r = H, W

        kv = self.kv(x_kv)
        k, v = kv.chunk(2, dim=-1)

        k = k.unsqueeze(1).expand(-1, self.num_heads, -1, -1).reshape(B * self.num_heads, -1, self.head_dim)
        v = v.unsqueeze(1).expand(-1, self.num_heads, -1, -1).reshape(B * self.num_heads, -1, self.head_dim)
        q = q.flatten(0, 1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_out = (attn @ v).reshape(B, self.num_heads, N, self.head_dim).transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out


# ---------- ABlock_MobileMQA（自动计算num_heads，与原ABlock兼容） ----------
class ABlock_MobileMQA(nn.Module):
    """
    Area-attention block with Mobile MQA.
    num_heads 默认根据 dim // 32 自动计算，与原 ABlock 保持一致。
    """
    def __init__(self, dim: int, num_heads: int = None, mlp_ratio: float = 1.2, area: int = 1, sr_ratio: int = 2):
        super().__init__()
        if num_heads is None:
            num_heads = dim // 32   # 与原 ABlock 的 head 数计算方式一致
        self.attn = MobileMQA(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_attn_flat = self.attn(x_flat, H, W)
        x_attn = x_attn_flat.transpose(1, 2).reshape(B, C, H, W)
        x = x + x_attn
        return x + self.mlp(x)


# ---------- A2C2f_MobileMQA（参数顺序与原A2C2f完全兼容） ----------
class A2C2f_MobileMQA(nn.Module):
    """
    A2C2f module with optional Mobile MQA integration.
    构造函数参数顺序与原 A2C2f 一致，可直接替换 YAML 中的 A2C2f。
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
        use_mqa: bool = True,            # 是否启用 Mobile MQA（默认启用）
        mqa_sr_ratio: int = 2,            # Mobile MQA 空间下采样率
    ):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList()
        for _ in range(n):
            if a2:
                if use_mqa:
                    # 使用 Mobile MQA 增强的 ABlock（堆叠两个，自动计算 num_heads）
                    block = nn.Sequential(
                        ABlock_MobileMQA(c_, mlp_ratio=mlp_ratio, area=area, sr_ratio=mqa_sr_ratio),
                        ABlock_MobileMQA(c_, mlp_ratio=mlp_ratio, area=area, sr_ratio=mqa_sr_ratio)
                    )
                else:
                    # 使用原始 ABlock（需要已定义）
                    from .block import ABlock
                    block = nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            else:
                # 使用 C3k（需要已定义）
                from .block import C3k
                block = C3k(c_, c_, 2, shortcut, g)
            self.m.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
        return y

if __name__ == '__main__':
    model = A2C2f_MobileMQA(c1=128, c2=256, n=2, a2=True, area=4)
    x = torch.randn(1, 128, 32, 32)
    out = model(x)
    print(out.shape)  # 应输出 torch.Size([1, 256, 32, 32])