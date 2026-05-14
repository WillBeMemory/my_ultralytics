import torch
import torch.nn as nn
import torch.nn.functional as F


# ================== 标准 Bottleneck（与 ultralytics C3k2 中的 Bottleneck 一致） ==================
class Bottleneck(nn.Module):
    """标准 Bottleneck：1x1 降维 → 3x3 卷积 → 1x1 升维，可选残差连接"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # 中间通道数
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


# ================== 背景填充模块（软填充） ==================
class AdaptiveBackgroundFill(nn.Module):
    """
    自适应背景填充：利用每个通道的全局最小值作为基准，
    对局部对比度低的区域进行软填充，抑制均匀背景噪声。
    """
    def __init__(self, ch, pool_size=3, fill_strength=0.8, bg_thresh_ratio=0.5):
        super().__init__()
        self.ch = ch
        self.pool_size = pool_size
        self.fill_strength = fill_strength
        self.bg_thresh_ratio = bg_thresh_ratio

    def forward(self, x):
        B, C, H, W = x.shape
        # 基准值：每个通道的全局最小值
        baseline = x.view(B, C, -1).min(dim=-1)[0].view(B, C, 1, 1)

        # 局部对比度：max - min
        pad = self.pool_size // 2
        max_s = F.max_pool2d(x, self.pool_size, stride=1, padding=pad)
        min_s = -F.max_pool2d(-x, self.pool_size, stride=1, padding=pad)
        local_contrast = max_s - min_s

        # 动态阈值：低于通道平均对比度的一定比例视为背景
        mean_contrast = local_contrast.mean(dim=[2, 3], keepdim=True) + 1e-6
        bg_mask = (local_contrast < self.bg_thresh_ratio * mean_contrast).float()

        # 软填充
        out = x * (1 - self.fill_strength * bg_mask) + baseline * self.fill_strength * bg_mask
        return out


# ================== 通道感知边缘增强（注意力部分） ==================
class ChannelAwareEdgeEnhance_Attn(nn.Module):
    """仅执行通道筛选 + 空间边缘增强，不改变通道数，不包含卷积精炼"""
    def __init__(self, ch, pool_size=3, ch_sharp=5.0, ch_thresh=0.5, edge_sharp=5.0, edge_thresh=0.5):
        super().__init__()
        self.ch = ch
        self.pool_size = pool_size
        # 固定超参数 (requires_grad=False)
        self.ch_sharp = nn.Parameter(torch.tensor(ch_sharp), requires_grad=False)
        self.ch_thresh = nn.Parameter(torch.tensor(ch_thresh), requires_grad=False)
        self.edge_sharp = nn.Parameter(torch.tensor(edge_sharp), requires_grad=False)
        self.edge_thresh = nn.Parameter(torch.tensor(edge_thresh), requires_grad=False)

    def forward(self, x):
        # 根据输入 x 的 dtype 转换超参数（适配 AMP 混合精度训练）
        ch_sharp = self.ch_sharp.to(dtype=x.dtype)
        ch_thresh = self.ch_thresh.to(dtype=x.dtype)
        edge_sharp = self.edge_sharp.to(dtype=x.dtype)
        edge_thresh = self.edge_thresh.to(dtype=x.dtype)

        B, C, H, W = x.shape
        pad = self.pool_size // 2
        x_abs = x.abs()

        # 通道权重：max - avg 差异
        max_ch = F.adaptive_max_pool2d(x_abs, 1)
        avg_ch = F.adaptive_avg_pool2d(x_abs, 1)
        diff_ch = max_ch - avg_ch
        ch_weight = torch.sigmoid(ch_sharp * (diff_ch - ch_thresh))

        # 空间边缘权重：max - min
        x_spatial = x_abs.mean(dim=1, keepdim=True)
        max_s = F.max_pool2d(x_spatial, self.pool_size, stride=1, padding=pad)
        min_s = -F.max_pool2d(-x_spatial, self.pool_size, stride=1, padding=pad)
        edge = max_s - min_s
        edge_weight = torch.sigmoid(edge_sharp * (edge - edge_thresh))

        # 应用增强
        out = x * ch_weight
        out = out * (1.0 + edge_weight)
        return out


# ================== 完整模块：SoftFillEdgeEnhance ==================
class SoftFillEdgeEnhance(nn.Module):
    """
    背景软填充 + 通道‑空间注意力增强 + Bottleneck 精炼

    YOLO 参数顺序:
    c1, c2, n, pool_size, fill_strength, bg_thresh_ratio,
    ch_sharp, ch_thresh, edge_sharp, edge_thresh,
    bottleneck_e, bottleneck_shortcut
    """
    def __init__(self, c1, c2, n=1, pool_size=3,
                 fill_strength=0.8, bg_thresh_ratio=0.5,
                 ch_sharp=5.0, ch_thresh=0.5,
                 edge_sharp=5.0, edge_thresh=0.5,
                 bottleneck_e=0.5, bottleneck_shortcut=True):
        super().__init__()
        self.bg_fill = AdaptiveBackgroundFill(c1, pool_size, fill_strength, bg_thresh_ratio)
        self.attn = ChannelAwareEdgeEnhance_Attn(c1, pool_size, ch_sharp, ch_thresh, edge_sharp, edge_thresh)

        # Bottleneck 序列：n 个串联的 Bottleneck，输入输出通道均为 c1
        self.bottlenecks = nn.Sequential(*[
            Bottleneck(c1, c1, shortcut=bottleneck_shortcut, e=bottleneck_e)
            for _ in range(n)
        ])

        # 输出投影（当 c2 != c1 时必需）
        self.proj = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

    def forward(self, x):
        # 关键修复：将输入 x 对齐到模块内部权重的 dtype（消除 AMP 下的类型不匹配）
        # 获取第一个可学习参数的 dtype（如 bottlenecks[0].cv1.weight）
        if len(self.bottlenecks) > 0:
            target_dtype = self.bottlenecks[0].cv1.weight.dtype
        else:
            target_dtype = x.dtype
        x = x.to(target_dtype)

        out = self.bg_fill(x)        # 1. 背景填充
        out = self.attn(out)         # 2. 注意力增强（通道筛选+边缘增强）
        out = self.bottlenecks(out)  # 3. 特征精炼（n 个 Bottleneck）
        out = self.proj(out)         # 4. 通道对齐
        return out


# ================== 测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 模拟 P5 特征图
    batch, C, H, W = 2, 256, 20, 20
    x = torch.randn(batch, C, H, W).to(device)

    # 实例化模块，n=2 表示两次 Bottleneck 精炼
    model = SoftFillEdgeEnhance(C, 256, n=2, pool_size=3,
                                fill_strength=0.8, bg_thresh_ratio=0.5,
                                ch_sharp=5.0, ch_thresh=0.5,
                                edge_sharp=5.0, edge_thresh=0.5,
                                bottleneck_e=0.5, bottleneck_shortcut=True).to(device)
    print(model)

    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")

    loss = out.mean()
    loss.backward()
    print("Backward pass success.")