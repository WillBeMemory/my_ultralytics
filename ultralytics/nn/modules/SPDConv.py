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


class SPDConv(nn.Module):
    """
    SPDConv with edge attention and grouped channel mixing.

    Args:
        c1 (int): 输入通道数。
        c2 (int): 输出通道数。
        k (int): 压缩卷积核大小。默认 1。
        s (int): 下采样步长。2 启用 SPD + 边缘注意力 + 分组混洗；1 则退化为普通卷积。
        g (int): 边缘注意力的分组数（用于计算 Max-Avg 差异）。
        act (bool): 是否使用 SiLU 激活函数。默认 True。
    """

    def __init__(self, c1, c2, k=1, s=2, g=4, act=True):
        super().__init__()
        self.s = s
        self.c1 = c1
        self.g = g
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

        if s == 2:
            c_mid = c1 * 4  # SPD 膨胀后的通道数

            # 轻量边缘注意力（1×1 卷积）
            self.edge_attn = nn.Sequential(
                nn.Conv2d(c_mid, c_mid // 4, 1, bias=False),
                nn.BatchNorm2d(c_mid // 4),
                nn.SiLU(inplace=True),
                nn.Conv2d(c_mid // 4, c_mid, 1, bias=False),
                nn.BatchNorm2d(c_mid),
                nn.Sigmoid(),
            )

            # 分组卷积：每 4 个通道一组（groups=c1），让来自同一 2×2 邻域的通道交互
            self.group_mixer = nn.Conv2d(c_mid, c_mid, kernel_size=1, groups=c1, bias=False)
            self.group_bn = nn.BatchNorm2d(c_mid)
            self.group_act = nn.SiLU(inplace=True)

            # 压缩卷积（stride=1，降维到 c2）
            self.conv = nn.Conv2d(c_mid, c2, k, stride=1, padding=autopad(k), bias=False)
            self.bn = nn.BatchNorm2d(c2)

        else:
            # 普通卷积（无 SPD）
            self.conv = nn.Conv2d(c1, c2, k, s, padding=autopad(k), bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.edge_attn = None
            self.group_mixer = None

    def forward(self, x):
        if self.s == 2:
            B, C, H, W = x.shape

            # ---------- SPD 层：空间到深度 ----------
            pad_h = (2 - H % 2) % 2
            pad_w = (2 - W % 2) % 2
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            x_spd = F.pixel_unshuffle(x, downscale_factor=2)  # (B, 4C, H/2, W/2)

            # ---------- 边缘注意力 ----------
            C_mid = x_spd.shape[1]  # 4*C
            C_per_grp = C_mid // self.g
            x_grp = x_spd.view(B, self.g, C_per_grp, x_spd.shape[2], x_spd.shape[3])
            grp_max = x_grp.max(dim=2, keepdim=False)[0]
            grp_avg = x_grp.mean(dim=2, keepdim=False)
            grp_edge = grp_max - grp_avg
            grp_edge_up = grp_edge.repeat_interleave(C_per_grp, dim=1)
            edge_weight = self.edge_attn(grp_edge_up)
            x_edge = x_spd * edge_weight + x_spd  # 残差连接

            # ---------- 分组混洗（每4通道交互） ----------
            x_group = self.group_act(self.group_bn(self.group_mixer(x_edge)))

            # ---------- 压缩卷积 ----------
            out = self.act(self.bn(self.conv(x_group)))
            return out

        else:
            # 普通卷积（无下采样）
            return self.act(self.bn(self.conv(x)))


# ---------- 简单 main 测试 ----------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 测试下采样模式
    model1 = SPDConv(c1=64, c2=128, k=3, s=2).to(device)
    model1.train()
    x1 = torch.randn(2, 64, 32, 32).to(device)
    y1 = model1(x1)
    print(f"SPD mode input: {x1.shape} -> output: {y1.shape} (expected: [2,128,16,16])")
    loss1 = y1.mean()
    loss1.backward()
    print("Gradients OK for SPD mode")

    # 测试普通卷积模式
    model2 = SPDConv(c1=64, c2=64, k=3, s=1).to(device)
    model2.train()
    x2 = torch.randn(2, 64, 32, 32).to(device)
    y2 = model2(x2)
    print(f"Normal mode input: {x2.shape} -> output: {y2.shape} (expected: [2,64,32,32])")
    loss2 = y2.mean()
    loss2.backward()
    print("Gradients OK for normal mode")