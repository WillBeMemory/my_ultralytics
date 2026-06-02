import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================== 原有子模块 ==================
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
            out = out + identity      # 残差后无激活（与您之前一致）
        return out


class AdaptiveBackgroundFill(nn.Module):
    def __init__(self, ch, pool_size=3, fill_strength=0.8, bg_thresh_ratio=0.5):
        super().__init__()
        assert pool_size % 2 == 1, "pool_size must be odd"
        self.ch = ch
        self.pool_size = pool_size
        self.fill_strength = fill_strength
        self.bg_thresh_ratio = bg_thresh_ratio
        self.register_buffer('eps', torch.tensor(1e-6))

    def forward(self, x):
        B, C, H, W = x.shape
        baseline = x.view(B, C, -1).min(dim=-1)[0].view(B, C, 1, 1)
        pad = self.pool_size // 2
        max_s = F.max_pool2d(x, self.pool_size, stride=1, padding=pad)
        min_s = -F.max_pool2d(-x, self.pool_size, stride=1, padding=pad)
        local_contrast = max_s - min_s
        mean_contrast = local_contrast.mean(dim=[2, 3], keepdim=True) + self.eps
        bg_mask = (local_contrast < self.bg_thresh_ratio * mean_contrast).to(dtype=x.dtype)
        out = x * (1 - self.fill_strength * bg_mask) + baseline * self.fill_strength * bg_mask
        return out


class ChannelAwareEdgeEnhance_Attn(nn.Module):
    def __init__(self, ch, pool_size=3, ch_sharp=5.0, ch_thresh=0.5, edge_sharp=5.0, edge_thresh=0.5):
        super().__init__()
        assert pool_size % 2 == 1, "pool_size must be odd"
        self.ch = ch
        self.pool_size = pool_size
        # Learnable sharpness (init = given value, trainable)
        self.log_ch_sharp = nn.Parameter(torch.tensor(math.log(max(ch_sharp, 1e-3))))
        self.log_edge_sharp = nn.Parameter(torch.tensor(math.log(max(edge_sharp, 1e-3))))
        self.ch_thresh = nn.Parameter(torch.tensor(ch_thresh))
        self.edge_thresh = nn.Parameter(torch.tensor(edge_thresh))
        # Warmup steps (sharpness starts low to avoid early binarization)
        self.register_buffer('_step', torch.tensor(0, dtype=torch.long))
        self.warmup_steps = 2000
        self.register_buffer('one', torch.tensor(1.0))

    def forward(self, x):
        dtype = x.dtype
        device = x.device

        # Warmup: sharpness starts low, ramps up (prevents early binarization)
        self._step += 1
        warmup_alpha = min(1.0, self._step.float() / self.warmup_steps)
        ch_sharp = torch.exp(self.log_ch_sharp) * warmup_alpha
        ch_thresh = self.ch_thresh
        edge_sharp = torch.exp(self.log_edge_sharp) * warmup_alpha
        edge_thresh = self.edge_thresh

        B, C, H, W = x.shape
        pad = self.pool_size // 2
        x_abs = x.abs()

        max_ch = x_abs.view(B, C, -1).max(dim=-1, keepdim=True)[0].view(B, C, 1, 1)
        avg_ch = F.adaptive_avg_pool2d(x_abs, 1)
        diff_ch = max_ch - avg_ch
        ch_weight = torch.sigmoid(ch_sharp * (diff_ch - ch_thresh))

        x_spatial = x_abs.mean(dim=1, keepdim=True)
        max_s = F.max_pool2d(x_spatial, self.pool_size, stride=1, padding=pad)
        min_s = -F.max_pool2d(-x_spatial, self.pool_size, stride=1, padding=pad)
        edge = max_s - min_s
        edge_weight = torch.sigmoid(edge_sharp * (edge - edge_thresh))

        out = x * ch_weight
        out = out * (self.one.to(dtype) + edge_weight)
        return out


# ================== 改进后的 SoftFillEdgeEnhance（末尾追加深度可分离卷积） ==================
class SoftFillEdgeEnhance(nn.Module):
    def __init__(self, c1, c2, n=1, pool_size=3, bottleneck_e=0.5,
                 fill_strength=0.8, bg_thresh_ratio=0.5,
                 ch_sharp=5.0, ch_thresh=0.5,
                 edge_sharp=5.0, edge_thresh=0.5,
                 bottleneck_shortcut=True,
                 use_dwconv=True):               # 新增：是否追加深度可分离卷积
        super().__init__()
        assert pool_size % 2 == 1, "pool_size must be odd"

        self.bg_fill = AdaptiveBackgroundFill(c1, pool_size, fill_strength, bg_thresh_ratio)
        self.attn = ChannelAwareEdgeEnhance_Attn(c1, pool_size, ch_sharp, ch_thresh, edge_sharp, edge_thresh)

        self.bottlenecks = nn.Sequential(*[
            Bottleneck(c1, c1, shortcut=bottleneck_shortcut, e=bottleneck_e)
            for _ in range(n)
        ])

        # 投影层（无 BN，无激活）
        if c1 != c2:
            self.proj = nn.Conv2d(c1, c2, 1, bias=False)
        else:
            self.proj = nn.Identity()

        # 末尾深度可分离卷积（输入输出通道均为 c2）
        if use_dwconv:
            self.dwconv = nn.Sequential(
                # Depthwise 3x3
                nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(inplace=True),
                # Pointwise 1x1
                nn.Conv2d(c2, c2, 1, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(inplace=True)
            )
        else:
            self.dwconv = nn.Identity()

    def forward(self, x):
        target_dtype = self.bottlenecks[0].cv1.weight.dtype if len(self.bottlenecks) > 0 else x.dtype
        x = x.to(target_dtype)

        out = self.bg_fill(x)
        out = self.attn(out)
        out = self.bottlenecks(out)
        out = self.proj(out)
        out = self.dwconv(out)   # 追加深度可分离精炼
        return out


# ================== 简单测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 通道改变的情况
    x = torch.randn(2, 128, 80, 80).to(device)
    model = SoftFillEdgeEnhance(128, 256, n=1, pool_size=3, fill_strength=0.8, bg_thresh_ratio=0.5,
                                ch_sharp=5.0, ch_thresh=0.5, edge_sharp=5.0, edge_thresh=0.5,
                                bottleneck_shortcut=True, use_dwconv=True).to(device)
    model.train()
    y = model(x)
    print(f"Input: {x.shape} → Output: {y.shape} (expected [2,256,80,80])")
    loss = y.mean()
    loss.backward()
    print("Gradients OK")