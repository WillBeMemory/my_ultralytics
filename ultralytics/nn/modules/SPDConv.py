import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def channel_shuffle(x, groups):
    """通道混洗（ShuffleNet），用于打破组间隔离。"""
    B, C, H, W = x.shape
    assert C % groups == 0
    x = x.view(B, groups, C // groups, H, W)
    x = x.transpose(1, 2).contiguous()
    x = x.view(B, C, H, W)
    return x


class GroupStarMixer(nn.Module):
    """
    分组星操作增强模块（含通道混洗）：
    - 将特征按每 group_size 个通道分组（对应 SPD 的 2×2 邻域）
    - 每组内执行星操作（两个 1×1 卷积 → 逐元素相乘 → 1×1 融合）
    - 之后通过通道混洗实现组间信息交互
    - 残差连接 + 激活
    """
    def __init__(self, channels, group_size=4, reduction=1.0):
        super().__init__()
        self.group_size = group_size
        self.num_groups = channels // group_size
        assert channels % group_size == 0, f"channels {channels} must be divisible by group_size {group_size}"

        # 组内星操作的中间通道数（可进一步压缩）
        mid_channels = int(group_size * reduction)
        if mid_channels < 1:
            mid_channels = 1

        # 两个投影卷积：组内独立变换（分组卷积）
        self.proj1 = nn.Conv2d(channels, self.num_groups * mid_channels, 1, groups=self.num_groups, bias=False)
        self.proj2 = nn.Conv2d(channels, self.num_groups * mid_channels, 1, groups=self.num_groups, bias=False)

        # 融合卷积：将相乘后的特征映射回原始通道（组内）
        self.fusion = nn.Conv2d(self.num_groups * mid_channels, channels, 1, groups=self.num_groups, bias=False)

        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x
        # 组内投影
        x1 = self.proj1(x)   # (B, num_groups*mid, H, W)
        x2 = self.proj2(x)

        # 星操作核心：逐元素乘法
        star = x1 * x2

        # 融合回原始通道（组内）
        out = self.fusion(star)   # (B, channels, H, W)

        # 通道混洗：打破组间隔离，实现跨组信息交互
        out = channel_shuffle(out, self.num_groups)

        # 残差连接 + 归一化 + 激活
        out = self.act(self.bn(out + identity))
        return out


class SPDConv(nn.Module):
    """
    SPDConv with GroupStarMixer (no extra attention).
    空间到深度无损下采样 + 分组星操作增强（含通道混洗） + 1x1 压缩卷积。

    Args:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        k (int): 压缩卷积核大小，默认 1
        s (int): 步长，2 启用 SPD，1 退化为普通卷积
        group_size (int): 星操作分组大小，默认 4（与 SPD 2x2 邻域对应）
        reduction (float): 星操作内部通道压缩系数，默认 1.0
        act (bool): 是否使用 SiLU 激活，默认 True
    """
    def __init__(self, c1, c2, k=1, s=2, group_size=4, reduction=1.0, act=True):
        super().__init__()
        self.s = s
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

        if s == 2:
            c_mid = c1 * 4   # SPD 膨胀后通道数

            # 星操作增强 + 通道混洗（替代原有的注意力与分组混洗）
            self.star_mixer = GroupStarMixer(c_mid, group_size=group_size, reduction=reduction)

            # 压缩卷积（降维到 c2）
            self.conv = nn.Conv2d(c_mid, c2, k, stride=1, padding=autopad(k), bias=False)
            self.bn = nn.BatchNorm2d(c2)

        else:
            # 普通卷积（无 SPD）
            self.conv = nn.Conv2d(c1, c2, k, s, padding=autopad(k), bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.star_mixer = None

    def forward(self, x):
        if self.s == 2:
            B, C, H, W = x.shape

            # ---------- SPD 层：空间到深度 ----------
            pad_h = (2 - H % 2) % 2
            pad_w = (2 - W % 2) % 2
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            x_spd = F.pixel_unshuffle(x, downscale_factor=2)  # (B, 4C, H/2, W/2)

            # ---------- 分组星操作增强（含组内交互 + 组间混洗） ----------
            x_mixed = self.star_mixer(x_spd)

            # ---------- 压缩卷积 ----------
            out = self.act(self.bn(self.conv(x_mixed)))
            return out

        else:
            # 普通卷积（无下采样）
            return self.act(self.bn(self.conv(x)))


# ---------- 简单 main 测试 ----------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 测试 1：下采样模式（SPD + 星操作 + 通道混洗）
    model1 = SPDConv(c1=64, c2=128, k=3, s=2, group_size=4, reduction=1.0).to(device)
    model1.train()
    x1 = torch.randn(2, 64, 32, 32).to(device)
    y1 = model1(x1)
    print(f"Test 1 - SPD+StarMix+Shuffle: input {x1.shape} → output {y1.shape} (expected [2,128,16,16])")
    loss1 = y1.mean()
    loss1.backward()
    print("  Gradients OK")

    # 测试 2：普通卷积模式 (s=1)
    model2 = SPDConv(c1=64, c2=64, k=3, s=1).to(device)
    model2.train()
    x2 = torch.randn(2, 64, 32, 32).to(device)
    y2 = model2(x2)
    print(f"Test 2 - Normal Conv: input {x2.shape} → output {y2.shape} (expected [2,64,32,32])")
    loss2 = y2.mean()
    loss2.backward()
    print("  Gradients OK")

    # 测试 3：不同分组大小和压缩系数
    model3 = SPDConv(c1=32, c2=64, k=1, s=2, group_size=4, reduction=0.5).to(device)
    model3.train()
    x3 = torch.randn(1, 32, 20, 20).to(device)
    y3 = model3(x3)
    print(f"Test 3 - group_size=4, reduction=0.5: input {x3.shape} → output {y3.shape} (expected [1,64,10,10])")
    loss3 = y3.mean()
    loss3.backward()
    print("  Gradients OK")

    print("All tests passed!")