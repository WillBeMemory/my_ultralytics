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


class GhostConv(nn.Module):
    """Ghost 卷积，ratio 控制内在/廉价通道比例，dw_groups 控制廉价操作的分组数（默认深度可分离）"""
    def __init__(self, c1, c2, kernel_size=1, stride=1, ratio=4, dw_groups=None, act=True):
        super().__init__()
        self.c_out_int = c2 // ratio
        self.c_out_cheap = c2 - self.c_out_int
        if dw_groups is None:
            dw_groups = self.c_out_int  # 深度可分离
        # 内在特征图
        self.primary_conv = nn.Sequential(
            nn.Conv2d(c1, self.c_out_int, kernel_size, stride, autopad(kernel_size), bias=False),
            nn.BatchNorm2d(self.c_out_int)
        )
        # 廉价操作
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(self.c_out_int, self.c_out_cheap, kernel_size=3, stride=1,
                      padding=1, groups=dw_groups, bias=False),
            nn.BatchNorm2d(self.c_out_cheap)
        )
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return self.act(torch.cat([x1, x2], dim=1))


class SPDConv(nn.Module):
    """
    SPDConv（分组卷积版）：
    - 保留 SPD + 轻量分组卷积（3×3，groups=4）+ GhostConv 压缩
    - 分组数小，碎片化低，计算量接近普通 3×3 stride-2 卷积
    - 同时保留通道间的局部交互和无损下采样
    """
    def __init__(self, c1, c2, k=1, s=2, groups=16, ghost_ratio=4, dw_groups=None, act=True):
        super().__init__()
        self.s = s
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
        if s == 2:
            c_mid = c1 * 4
            # 轻量分组卷积（3×3，groups=groups，推荐4或8）
            self.group_conv = nn.Sequential(
                nn.Conv2d(c_mid, c_mid, kernel_size=3, stride=1, padding=1,
                          groups=groups, bias=False),
                nn.BatchNorm2d(c_mid),
                nn.SiLU(inplace=True)
            )
            # GhostConv 压缩
            self.compress = GhostConv(c_mid, c2, kernel_size=k, stride=1,
                                      ratio=ghost_ratio, dw_groups=dw_groups, act=False)
            self.bn = nn.BatchNorm2d(c2)
        else:
            # 普通下采样：GhostConv + stride
            self.conv = GhostConv(c1, c2, kernel_size=k, stride=s,
                                  ratio=ghost_ratio, dw_groups=dw_groups, act=False)
            self.bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        if self.s == 2:
            B, C, H, W = x.shape
            # 补偶
            pad_h = (2 - H % 2) % 2
            pad_w = (2 - W % 2) % 2
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            x_spd = F.pixel_unshuffle(x, downscale_factor=2)  # (B, 4C, H/2, W/2)
            x = self.group_conv(x_spd)
            x = self.compress(x)
            return self.act(self.bn(x))
        else:
            return self.act(self.bn(self.conv(x)))


# ---------- 测试 ----------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 测试1：groups=4，深度可分离廉价操作（默认）
    model1 = SPDConv(c1=64, c2=128, k=3, s=2, groups=4, ghost_ratio=4).to(device)
    model1.train()
    x1 = torch.randn(2, 64, 32, 32).to(device)
    y1 = model1(x1)
    print(f"Test1 (groups=4, dw cheap): {x1.shape} → {y1.shape} (expected [2,128,16,16])")
    loss1 = y1.mean()
    loss1.backward()
    print("  Gradients OK")

    # 测试2：groups=8，廉价操作为标准 3×3（dw_groups=1）
    model2 = SPDConv(c1=64, c2=128,groups=8, k=3, s=2, ghost_ratio=4, dw_groups=1).to(device)
    model2.train()
    x2 = torch.randn(2, 64, 32, 32).to(device)
    y2 = model2(x2)
    print(f"Test2 (groups=8, std cheap): {x2.shape} → {y2.shape} (expected [2,128,16,16])")
    loss2 = y2.mean()
    loss2.backward()
    print("  Gradients OK")

    # 测试3：普通卷积模式 (s=1)
    model3 = SPDConv(c1=64, c2=64, k=3, s=1, groups=4).to(device)
    model3.train()
    x3 = torch.randn(2, 64, 32, 32).to(device)
    y3 = model3(x3)
    print(f"Test3 (normal conv): {x3.shape} → {y3.shape} (expected [2,64,32,32])")
    loss3 = y3.mean()
    loss3.backward()
    print("  Gradients OK")