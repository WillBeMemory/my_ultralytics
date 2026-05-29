import torch
import torch.nn as nn
import torch.nn.functional as F


# ================== 修正后的 Bottleneck（残差相加后不再激活） ==================
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
            out = out + identity
        return out   # 不再加激活，与原YOLO风格一致


# ================== 背景软填充（不变） ==================
class AdaptiveBackgroundFill(nn.Module):
    def __init__(self, ch, pool_size=3, fill_strength=0.8, bg_thresh_ratio=0.5):
        super().__init__()
        assert pool_size % 2 == 1, "pool_size must be odd"
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


# ================== 通道感知边缘增强（修复类型兼容） ==================
class ChannelAwareEdgeEnhance_Attn(nn.Module):
    def __init__(self, ch, pool_size=3, ch_sharp=5.0, ch_thresh=0.5, edge_sharp=5.0, edge_thresh=0.5):
        super().__init__()
        assert pool_size % 2 == 1, "pool_size must be odd"
        self.ch = ch
        self.pool_size = pool_size
        # 使用 register_buffer 存储超参数，不参与梯度
        self.register_buffer('ch_sharp', torch.tensor(ch_sharp))
        self.register_buffer('ch_thresh', torch.tensor(ch_thresh))
        self.register_buffer('edge_sharp', torch.tensor(edge_sharp))
        self.register_buffer('edge_thresh', torch.tensor(edge_thresh))

    def forward(self, x):
        dtype = x.dtype
        device = x.device
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

        # 类型安全：确保常量 1 与输入 dtype 一致
        one = torch.tensor(1.0, dtype=dtype, device=device)
        out = x * ch_weight
        out = out * (one + edge_weight)
        return out


# ================== 优化后的 SoftFillEdgeEnhance（含残差、BN投影） ==================
class SoftFillEdgeEnhance(nn.Module):
    def __init__(self, c1, c2, n=1, pool_size=3, bottleneck_e=0.5,
                 fill_strength=0.8, bg_thresh_ratio=0.5,
                 ch_sharp=5.0, ch_thresh=0.5,
                 edge_sharp=5.0, edge_thresh=0.5,
                 bottleneck_shortcut=True, main_shortcut=True):
        super().__init__()
        assert pool_size % 2 == 1, "pool_size must be odd"
        self.main_shortcut = main_shortcut and c1 == c2

        self.bg_fill = AdaptiveBackgroundFill(c1, pool_size, fill_strength, bg_thresh_ratio)
        self.attn = ChannelAwareEdgeEnhance_Attn(c1, pool_size, ch_sharp, ch_thresh, edge_sharp, edge_thresh)

        self.bottlenecks = nn.Sequential(*[
            Bottleneck(c1, c1, shortcut=bottleneck_shortcut, e=bottleneck_e)
            for _ in range(n)
        ])

        # 投影层增加 BN，稳定训练
        if c1 != c2:
            self.proj = nn.Sequential(
                nn.Conv2d(c1, c2, 1, bias=False),
                nn.BatchNorm2d(c2)
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        # AMP 兼容：对齐 bottleneck 权重的 dtype
        target_dtype = self.bottlenecks[0].cv1.weight.dtype if len(self.bottlenecks) > 0 else x.dtype
        x = x.to(target_dtype)

        out = self.bg_fill(x)
        out = self.attn(out)
        out = self.bottlenecks(out)
        out = self.proj(out)

        if self.main_shortcut:
            out = out + x
        return out


# ================== 前向传播测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 测试 1: 通道不变，带残差
    model1 = SoftFillEdgeEnhance(c1=64, c2=64, n=1, pool_size=3,
                                 fill_strength=0.8, bg_thresh_ratio=0.5,
                                 ch_sharp=5.0, ch_thresh=0.5,
                                 edge_sharp=5.0, edge_thresh=0.5,
                                 bottleneck_e=0.5, bottleneck_shortcut=True,
                                 main_shortcut=True).to(device)
    model1.train()
    x1 = torch.randn(2, 64, 32, 32, dtype=torch.float16 if device == 'cuda' else torch.float32).to(device)
    y1 = model1(x1)
    print(f"Test1 (c1==c2) input: {x1.shape} → output: {y1.shape} (expected: [2,64,32,32])")
    loss1 = y1.mean()
    loss1.backward()
    print("Test1 backward passed.")

    # 测试 2: 通道改变，无残差
    model2 = SoftFillEdgeEnhance(c1=128, c2=256, n=2, pool_size=3,
                                 fill_strength=0.5, bg_thresh_ratio=0.5,
                                 ch_sharp=3.0, ch_thresh=0.5,
                                 edge_sharp=5.0, edge_thresh=0.3,
                                 bottleneck_e=0.5, bottleneck_shortcut=True,
                                 main_shortcut=False).to(device)
    model2.train()
    x2 = torch.randn(2, 128, 32, 32, dtype=torch.float16 if device == 'cuda' else torch.float32).to(device)
    y2 = model2(x2)
    print(f"Test2 (c1!=c2) input: {x2.shape} → output: {y2.shape} (expected: [2,256,32,32])")
    loss2 = y2.mean()
    loss2.backward()
    print("Test2 backward passed.")

    print("All tests passed!")