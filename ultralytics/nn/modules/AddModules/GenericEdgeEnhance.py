import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    """Standard bottleneck with residual + post-add SiLU (non-negative gate)."""
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
            out = self.act(out + identity)       # post-residual SiLU → non-negative gate
        else:
            out = self.act(out)
        return out


class AdaptiveBackgroundFill(nn.Module):
    """Soft background filling: pushes low-contrast regions toward channel minima."""

    def __init__(self, ch, pool_size=3, fill_strength=0.8, bg_thresh_ratio=0.5):
        super().__init__()
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
        return x * (1 - self.fill_strength * bg_mask) + baseline * self.fill_strength * bg_mask


class ChannelAwareEdgeEnhance(nn.Module):
    """Channel + spatial dual-path edge enhancement with learnable sharpness."""

    def __init__(self, ch, pool_size=3,
                 ch_sharp=5.0, ch_thresh=0.5,
                 edge_sharp=5.0, edge_thresh=0.5):
        super().__init__()
        self.ch = ch
        self.pool_size = pool_size
        self.log_ch_sharp = nn.Parameter(torch.tensor(math.log(max(ch_sharp, 1e-3))))
        self.log_edge_sharp = nn.Parameter(torch.tensor(math.log(max(edge_sharp, 1e-3))))
        self.ch_thresh = nn.Parameter(torch.tensor(ch_thresh))
        self.edge_thresh = nn.Parameter(torch.tensor(edge_thresh))
        self.register_buffer('_step', torch.tensor(0, dtype=torch.long))
        self.warmup_steps = 2000

    def forward(self, x):
        self._step += 1
        warmup_alpha = min(1.0, self._step.float() / self.warmup_steps)
        ch_sharp = torch.exp(self.log_ch_sharp) * warmup_alpha
        edge_sharp = torch.exp(self.log_edge_sharp) * warmup_alpha

        B, C, H, W = x.shape
        pad = self.pool_size // 2
        x_abs = x.abs()

        # Channel weight
        max_ch = x_abs.view(B, C, -1).max(dim=-1, keepdim=True)[0].view(B, C, 1, 1)
        avg_ch = F.adaptive_avg_pool2d(x_abs, 1)
        ch_weight = torch.sigmoid(ch_sharp * (max_ch - avg_ch - self.ch_thresh))

        # Spatial edge weight
        x_spatial = x_abs.mean(dim=1, keepdim=True)
        max_s = F.max_pool2d(x_spatial, self.pool_size, stride=1, padding=pad)
        min_s = -F.max_pool2d(-x_spatial, self.pool_size, stride=1, padding=pad)
        edge = max_s - min_s
        edge_weight = torch.sigmoid(edge_sharp * (edge - self.edge_thresh))

        return x * ch_weight * (1.0 + edge_weight)


class GenericEdgeEnhance(nn.Module):
    """Resolution-adaptive soft-fill + edge enhancement block.

    Auto-adjusts aggressiveness based on feature map resolution:
      Large H×W (>100): gentle (preserve details for P2/P3)
      Medium H×W (50~100): moderate (P3/P4)
      Small H×W (<50): aggressive (clean semantics for P4/P5)

    Drop-in replacement for C3k2 at any backbone/neck position.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of bottleneck blocks.
        adaptive (bool): Enable resolution-adaptive parameter scaling.
        pool_size (int): Local contrast estimation window.
        bottleneck_e (float): Bottleneck expansion ratio.
        bottleneck_shortcut (bool): Residual connection in bottleneck.
        base_sharp (float): Base sharpness for sigmoid gate (adptively scaled).
        base_fill (float): Base fill strength for background suppression.
        residual_gate (bool): Learnable blend between enhanced and original.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,            # YAML convention (ignored)
        e: float = 0.5,               # YAML convention (bottleneck expansion)
        adaptive: bool = True,
        pool_size: int = 3,
        bottleneck_shortcut: bool = True,
        base_sharp: float = 5.0,
        base_fill: float = 0.8,
        residual_gate: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert pool_size % 2 == 1

        self.adaptive = adaptive
        self.residual_gate = residual_gate

        # Static base parameters
        self.base_sharp = base_sharp
        self.base_fill = base_fill
        self.bg_thresh = 0.5

        # Resolution-adaptive scaling (learnable per-instance)
        if adaptive:
            # log_res_bias controls the resolution transition point
            # high → prefers detail-preserving mode at larger resolutions
            self.log_res_bias = nn.Parameter(torch.tensor(10.0))  # ≈ ln(22026)
        else:
            self.register_buffer('log_res_bias', torch.tensor(10.0))

        # Core sub-modules
        self.bg_fill = AdaptiveBackgroundFill(c1, pool_size, base_fill, self.bg_thresh)
        self.edge_attn = ChannelAwareEdgeEnhance(c1, pool_size,
                                                  ch_sharp=base_sharp, ch_thresh=0.5,
                                                  edge_sharp=base_sharp, edge_thresh=0.5)

        self.bottlenecks = nn.Sequential(*[
            Bottleneck(c1, c1, shortcut=bottleneck_shortcut, e=e)
            for _ in range(n)
        ])

        # Channel projection
        self.proj = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

        # Learnable residual gate: out = α * enhanced + (1-α) * original
        if residual_gate:
            self.log_alpha = nn.Parameter(torch.tensor(0.0))  # init α=0.5

    def _resolution_scale(self, H, W):
        """Compute auto-scale factor from spatial resolution.

        scale → 0 for large H×W (detail layer: soft enhancement)
        scale → 1 for small H×W (semantic layer: aggressive enhancement)
        """
        res = H * W
        # sigmoid switch: res >> threshold → scale → 0
        threshold = torch.exp(self.log_res_bias)
        scale = torch.sigmoid((torch.log(threshold) - torch.log(torch.tensor(res, dtype=torch.float32, device=self.log_res_bias.device))))
        return scale.item()

    def forward(self, x):
        target_dtype = self.bottlenecks[0].cv1.weight.dtype if len(self.bottlenecks) > 0 else x.dtype
        x = x.to(target_dtype)

        H, W = x.shape[2], x.shape[3]
        res_scale = self._resolution_scale(H, W) if self.adaptive else 0.5

        # Scale fill strength by resolution
        self.bg_fill.fill_strength = self.base_fill * res_scale

        # Process
        out = self.bg_fill(x)
        out = self.edge_attn(out)
        out = self.bottlenecks(out)
        out = self.proj(out)

        # Optional learned residual blend
        if self.residual_gate:
            alpha = torch.sigmoid(self.log_alpha)
            out = alpha * out + (1 - alpha) * x

        return out


# ================== Test ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("=" * 60)

    for hw in [(160, 160), (80, 80), (40, 40), (20, 20)]:
        x = torch.randn(2, 128, hw[0], hw[1], device=device)
        model = GenericEdgeEnhance(128, 256, n=1, adaptive=True).to(device)
        model.train()
        y = model(x)
        scale = model._resolution_scale(hw[0], hw[1])
        print(f"{hw[0]}×{hw[1]}: {x.shape} → {y.shape}  res_scale={scale:.3f}")
        loss = y.sum()
        loss.backward()
    print("✅ All tests passed")
