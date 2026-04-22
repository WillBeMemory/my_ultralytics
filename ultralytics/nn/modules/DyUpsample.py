# ============================================
# File: ultralytics/nn/modules/DySample.py
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    """
    轻量级动态上采样器 (DySample) —— 自适应通道版本。
    无需指定输入通道数，模块会在第一次前向传播时自动构建内部结构。

    参数:
        scale (int): 上采样尺度因子。默认为 2。
        style (str): 采样风格，'lp' 或 'pl'。默认为 'lp'。
        groups (int): 分组数，用于减少参数量。默认为 4。
        dyscope (bool): 是否启用动态范围因子。默认为 False。
    """
    def __init__(self, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        self.dyscope = dyscope
        assert style in ['lp', 'pl'], "style must be 'lp' or 'pl'"

        # 内部层将在第一次 forward 时构建
        self.offset = None
        self.scope = None
        self.in_channels = None
        self._initialized = False

        # 初始采样位置（与输入通道数无关）
        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        """生成初始采样位置（双线性初始化）"""
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def _initialize_layers(self, in_channels: int, device: torch.device, dtype: torch.dtype):
        """根据输入通道数动态创建卷积层"""
        if self._initialized and self.in_channels == in_channels:
            return  # 已初始化且通道数未变

        self.in_channels = in_channels
        groups = self.groups

        # 验证 'pl' 风格的通道数要求
        if self.style == 'pl':
            assert in_channels >= self.scale ** 2 and in_channels % self.scale ** 2 == 0, \
                f"For style 'pl', in_channels ({in_channels}) must be divisible by scale^2 ({self.scale**2})"
            actual_in = in_channels // self.scale ** 2
        else:
            actual_in = in_channels

        assert actual_in >= groups and actual_in % groups == 0, \
            f"Adjusted in_channels ({actual_in}) must be divisible by groups ({groups})"

        # 输出通道数
        if self.style == 'pl':
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * self.scale ** 2

        # 创建偏移量预测卷积
        self.offset = nn.Conv2d(actual_in, out_channels, 1).to(device=device, dtype=dtype)
        normal_init(self.offset, std=0.001)

        # 可选：动态范围因子
        if self.dyscope:
            self.scope = nn.Conv2d(actual_in, out_channels, 1, bias=False).to(device=device, dtype=dtype)
            constant_init(self.scope, val=0.)
        else:
            self.scope = None

        self._initialized = True

    def sample(self, x, offset):
        """根据偏移量从输入特征图上采样"""
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)

        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)

        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)

        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope') and self.scope is not None:
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope') and self.scope is not None:
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        # 首次前向时自动初始化
        if not self._initialized:
            self._initialize_layers(x.size(1), x.device, x.dtype)

        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 测试不同通道数的输入
    for ch in [64, 128, 256]:
        x = torch.randn(2, ch, 20, 20).to(device)
        model = DySample(scale=2, style='lp', groups=4).to(device)
        out = model(x)
        print(f"Input: {x.shape} -> Output: {out.shape}")
        loss = out.mean()
        loss.backward()
        print(f"  Backward OK, params: {sum(p.numel() for p in model.parameters()):,}")

    # 测试 'pl' 风格（需要通道数能被4整除）
    print("\nTesting 'pl' style:")
    x_pl = torch.randn(2, 64, 20, 20).to(device)
    model_pl = DySample(scale=2, style='pl', groups=4).to(device)
    out_pl = model_pl(x_pl)
    print(f"Input: {x_pl.shape} -> Output: {out_pl.shape}")
    loss = out_pl.mean()
    loss.backward()
    print("  Backward OK")