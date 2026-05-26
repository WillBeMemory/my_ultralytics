import torch
import torch.nn as nn
import torch.nn.functional as F


# ================== 标准 Bottleneck ==================
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, k[0], 1, padding=k[0] // 2, bias=False)
        self.cv2 = nn.Conv2d(c_, c2, k[1], 1, padding=k[1] // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.cv1(x)))
        out = self.bn2(self.cv2(out))
        if self.add:
            out = self.act(out + identity)
        else:
            out = self.act(out)
        return out


# ================== 背景软填充 ==================
class AdaptiveBackgroundFill(nn.Module):
    def __init__(self, ch, pool_size=3, fill_strength=0.8, bg_thresh_ratio=0.5):
        super().__init__()
        self.ch = ch
        self.pool_size = pool_size
        self.fill_strength = fill_strength
        self.bg_thresh_ratio = bg_thresh_ratio

    def forward(self, x):
        B, C, H, W = x.shape
        baseline = x.view(B, C, -1).min(dim=-1)[0].view(B, C, 1, 1)
        pad = self.pool_size // 2
        max_s = F.max_pool2d(x, self.pool_size, stride=1, padding=pad)
        min_s = -F.max_pool2d(-x, self.pool_size, stride=1, padding=pad)
        local_contrast = max_s - min_s
        eps = torch.tensor(1e-6, dtype=x.dtype, device=x.device)
        mean_contrast = local_contrast.mean(dim=[2, 3], keepdim=True) + eps
        bg_mask = (local_contrast < self.bg_thresh_ratio * mean_contrast).to(dtype=x.dtype)
        out = x * (1 - self.fill_strength * bg_mask) + baseline * self.fill_strength * bg_mask
        return out


# ================== 通道感知边缘增强 ==================
class ChannelAwareEdgeEnhance_Attn(nn.Module):
    def __init__(self, ch, pool_size=3, ch_sharp=5.0, ch_thresh=0.5, edge_sharp=5.0, edge_thresh=0.5):
        super().__init__()
        self.ch = ch
        self.pool_size = pool_size
        self.register_buffer('ch_sharp', torch.tensor(ch_sharp))
        self.register_buffer('ch_thresh', torch.tensor(ch_thresh))
        self.register_buffer('edge_sharp', torch.tensor(edge_sharp))
        self.register_buffer('edge_thresh', torch.tensor(edge_thresh))

    def forward(self, x):
        dtype = x.dtype
        ch_sharp = self.ch_sharp.to(dtype)
        ch_thresh = self.ch_thresh.to(dtype)
        edge_sharp = self.edge_sharp.to(dtype)
        edge_thresh = self.edge_thresh.to(dtype)

        B, C, H, W = x.shape
        pad = self.pool_size // 2
        x_abs = x.abs()

        # 通道权重
        max_ch = x_abs.view(B, C, -1).max(dim=-1, keepdim=True)[0].view(B, C, 1, 1)
        avg_ch = F.adaptive_avg_pool2d(x_abs, 1)
        diff_ch = max_ch - avg_ch
        ch_weight = torch.sigmoid(ch_sharp * (diff_ch - ch_thresh))

        # 空间边缘权重
        x_spatial = x_abs.mean(dim=1, keepdim=True)
        max_s = F.max_pool2d(x_spatial, self.pool_size, stride=1, padding=pad)
        min_s = -F.max_pool2d(-x_spatial, self.pool_size, stride=1, padding=pad)
        edge = max_s - min_s
        edge_weight = torch.sigmoid(edge_sharp * (edge - edge_thresh))

        out = x * ch_weight
        out = out * (1.0 + edge_weight)
        return out


# ================== 增强 Bottleneck（背景填充 + 边缘增强 + Bottleneck） ==================
class SoftFillEdgeBottleneck(nn.Module):
    def __init__(self, c, pool_size=3,
                 fill_strength=0.8, bg_thresh_ratio=0.5,
                 ch_sharp=5.0, ch_thresh=0.5,
                 edge_sharp=5.0, edge_thresh=0.5,
                 bottleneck_e=0.5, bottleneck_shortcut=True):
        super().__init__()
        self.bg_fill = AdaptiveBackgroundFill(c, pool_size, fill_strength, bg_thresh_ratio)
        self.attn = ChannelAwareEdgeEnhance_Attn(c, pool_size, ch_sharp, ch_thresh, edge_sharp, edge_thresh)
        self.bottleneck = Bottleneck(c, c, shortcut=bottleneck_shortcut, e=bottleneck_e)

    def forward(self, x):
        # 对齐 Bottleneck 权重的 dtype
        target_dtype = self.bottleneck.cv1.weight.dtype
        x = x.to(target_dtype)
        out = self.bg_fill(x)
        out = self.attn(out)
        out = self.bottleneck(out)
        return out


# ================== CSP 结构的 SoftFillEdgeEnhance（类似 C3k2） ==================
class SoftFillEdgeEnhance(nn.Module):
    """
    CSP 版本的 SoftFillEdgeEnhance，参考 C3k2 结构。
    参数：
        c1, c2: 输入/输出通道数
        n: 增强分支中 SoftFillEdgeBottleneck 重复次数
        e: CSP 隐藏通道扩展比（隐藏通道数 = int(c2 * e)）
        其他参数用于内部背景填充和边缘增强。
    """
    def __init__(self, c1, c2, n=1, e=0.5,
                 pool_size=3,
                 fill_strength=0.8, bg_thresh_ratio=0.5,
                 ch_sharp=5.0, ch_thresh=0.5,
                 edge_sharp=5.0, edge_thresh=0.5,
                 bottleneck_e=0.5, bottleneck_shortcut=True):
        super().__init__()
        self.c = int(c2 * e)          # 隐藏通道数
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, bias=False)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1, bias=False)

        # 增强分支（n 个串联的 SoftFillEdgeBottleneck）
        self.m = nn.ModuleList([
            SoftFillEdgeBottleneck(
                self.c, pool_size,
                fill_strength, bg_thresh_ratio,
                ch_sharp, ch_thresh, edge_sharp, edge_thresh,
                bottleneck_e, bottleneck_shortcut
            )
            for _ in range(n)
        ])

    def forward(self, x):
        # 对齐 cv1 权重的 dtype（AMP 兼容）
        target_dtype = self.cv1.weight.dtype
        x = x.to(target_dtype)

        # 1. 1x1 卷积分裂
        y = list(self.cv1(x).chunk(2, dim=1))   # y[0] 直接透传, y[1] 进入增强分支

        # 2. 增强分支依次执行，并将每个 block 的输出追加到列表（模拟 C2f 的梯度流）
        y.extend(m(y[-1]) for m in self.m)

        # 3. 拼接并投影
        return self.cv2(torch.cat(y, dim=1))

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 实例化模块：输入128通道，输出256通道，n=2个增强块，CSP扩展比e=0.5
    model = SoftFillEdgeEnhance(
        c1=128, c2=256, n=2, e=0.5,
        pool_size=3,
        fill_strength=0.8, bg_thresh_ratio=0.5,
        ch_sharp=5.0, ch_thresh=0.5,
        edge_sharp=5.0, edge_thresh=0.5,
        bottleneck_e=0.5, bottleneck_shortcut=True
    ).to(device)
    model.train()  # 设置为训练模式，使AMP兼容性测试有效

    # 随机输入：batch=2, channels=128, height=32, width=32
    x = torch.randn(2, 128, 32, 32).to(device)

    # 前向传播
    y = model(x)

    # 输出形状
    print(f"Input  shape: {x.shape}")  # (2, 128, 32, 32)
    print(f"Output shape: {y.shape}")  # 期望 (2, 256, 32, 32)

    # 反向传播测试
    loss = y.mean()
    loss.backward()
    print("Backward passed. Test OK!")