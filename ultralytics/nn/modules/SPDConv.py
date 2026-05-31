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


class ECA(nn.Module):
    """高效通道注意力（ECA）"""
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        y = x.mean(dim=(2, 3), keepdim=True)          # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)           # (B, 1, C)
        y = self.conv(y)                               # 一维卷积
        y = y.transpose(-1, -2).unsqueeze(-1)          # (B, C, 1, 1)
        return x * self.sigmoid(y)


class GhostConv(nn.Module):
    """Ghost 卷积（轻量通道生成）"""
    def __init__(self, c1, c2, kernel_size=1, stride=1, ratio=2, padding=None, dw_kernel_size=3, act=True):
        super().__init__()
        self.c_out_int = c2 // ratio
        self.c_out_cheap = c2 - self.c_out_int
        if padding is None:
            padding = (kernel_size - 1) // 2 if stride == 1 else kernel_size // 2

        # 标准卷积生成内在特征图
        self.primary_conv = nn.Sequential(
            nn.Conv2d(c1, self.c_out_int, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(self.c_out_int)
        )
        # 廉价操作生成额外特征图
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(self.c_out_int, self.c_out_int * (ratio - 1), dw_kernel_size, 1,
                      padding=(dw_kernel_size - 1) // 2, groups=self.c_out_int, bias=False),
            nn.BatchNorm2d(self.c_out_int * (ratio - 1))
        )
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return self.act(out)


class GroupStarMixer(nn.Module):
    """分组星操作 + 通道混洗（组内交互 → 组间交互）"""
    def __init__(self, channels, group_size=4, reduction=1.0):
        super().__init__()
        self.group_size = group_size
        self.num_groups = channels // group_size
        assert channels % group_size == 0

        mid_channels = int(group_size * reduction) or 1

        self.proj1 = nn.Conv2d(channels, self.num_groups * mid_channels, 1, groups=self.num_groups, bias=False)
        self.proj2 = nn.Conv2d(channels, self.num_groups * mid_channels, 1, groups=self.num_groups, bias=False)
        self.fusion = nn.Conv2d(self.num_groups * mid_channels, channels, 1, groups=self.num_groups, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x
        x1 = self.proj1(x)
        x2 = self.proj2(x)
        star = x1 * x2
        out = self.fusion(star)
        # 通道混洗
        out = channel_shuffle(out, self.num_groups)
        return self.act(self.bn(out + identity))


def channel_shuffle(x, groups):
    B, C, H, W = x.shape
    assert C % groups == 0
    x = x.view(B, groups, C // groups, H, W)
    x = x.transpose(1, 2).contiguous()
    return x.view(B, C, H, W)


class SPDConv(nn.Module):
    """
    SPDConv + GhostConv + ECA（轻量无损下采样，强化通道交互）
    """
    def __init__(self, c1, c2, k=1, s=2, ghost_ratio=2, act=True):
        super().__init__()
        self.s = s
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

        if s == 2:
            c_mid = c1 * 4
            # 分组星操作（含组间混洗）
            self.star_mixer = GroupStarMixer(c_mid, group_size=4, reduction=1.0)
            # 通道注意力
            self.eca = ECA(c_mid)
            # Ghost 卷积压缩（替代标准 1x1）
            self.compress = GhostConv(c_mid, c2, kernel_size=k, stride=1, ratio=ghost_ratio, act=False)
            self.bn = nn.BatchNorm2d(c2)
        else:
            self.conv = GhostConv(c1, c2, kernel_size=k, stride=s, ratio=ghost_ratio, act=False)
            self.bn = nn.BatchNorm2d(c2)
            self.star_mixer = None
            self.eca = None

    def forward(self, x):
        if self.s == 2:
            B, C, H, W = x.shape
            # 确保尺寸为偶数
            pad_h = (2 - H % 2) % 2
            pad_w = (2 - W % 2) % 2
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            x_spd = F.pixel_unshuffle(x, downscale_factor=2)
            # 分组星操作 + 混洗
            x = self.star_mixer(x_spd)
            # 通道注意力
            x = self.eca(x)
            # Ghost 卷积压缩
            x = self.compress(x)
            return self.act(self.bn(x))
        else:
            return self.act(self.bn(self.conv(x)))


# ---------- 测试 ----------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = SPDConv(c1=64, c2=128, k=3, s=2).to(device)
    model.train()
    x = torch.randn(2, 64, 32, 32).to(device)
    y = model(x)
    print(f"Input: {x.shape} → Output: {y.shape} (expected [2,128,16,16])")
    loss = y.mean()
    loss.backward()
    print("Gradients OK, test passed.")