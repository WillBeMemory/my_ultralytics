import torch
import torch.nn as nn
import torch.nn.functional as F


# ================== 辅助卷积块 ==================
def Conv2dBlock(in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False, act=True):
    """简单的 Conv2d + BN + SiLU 组合"""
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)]
    layers.append(nn.BatchNorm2d(out_ch))
    if act:
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


# ================== 标准 Bottleneck ==================
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=1.0):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv2dBlock(c1, c_, 1)
        self.cv2 = Conv2dBlock(c_, c2, 3, 1, 1, act=False)
        self.act = nn.SiLU(inplace=True)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        identity = x
        out = self.cv1(x)
        out = self.cv2(out)
        if self.add:
            out = self.act(out + identity)
        else:
            out = self.act(out)
        return out


# ================== 通道感知边缘增强（对齐 C3k2 计算量） ==================
class ChannelAwareEdgeEnhance(nn.Module):
    """
    通道感知边缘增强模块（类 C2f 结构，通过 e 控制分支通道数）

    流程：
    1. cv1: 1x1 卷积投影到 2*c (c = int(c2 * e))
    2. 通道分割：恒等分支 b + 增强分支 a
    3. 增强分支：双重注意力（通道筛选 + 空间边缘增强）→ 串行 Bottleneck 序列
    4. 拼接：恒等分支 b + 增强分支初始值 a + 每个 Bottleneck 输出
    5. cv2: 1x1 卷积融合到输出通道 c2

    YOLO 参数顺序（推荐）：
    [c2, n, e, pool_size, ch_sharp, ch_thresh, edge_sharp, edge_thresh, bottleneck_e, shortcut]
    n   : Bottleneck 个数
    e   : 分支通道比例（int(c2 * e)），控制计算量
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        e: float = 0.5,
        pool_size: int = 3,
        ch_sharp: float = 5.0,
        ch_thresh: float = 0.5,
        edge_sharp: float = 5.0,
        edge_thresh: float = 0.5,
        bottleneck_e: float = 1.0,
        shortcut: bool = True,
        **kwargs
    ):
        super().__init__()
        self.c = int(c2 * e)            # 分支通道数
        self.n = n
        self.pool_size = pool_size

        # 固定注意力参数
        self.register_buffer('ch_sharp', torch.tensor(ch_sharp))
        self.register_buffer('ch_thresh', torch.tensor(ch_thresh))
        self.register_buffer('edge_sharp', torch.tensor(edge_sharp))
        self.register_buffer('edge_thresh', torch.tensor(edge_thresh))

        # cv1: 1x1 投影到 2*c
        self.cv1 = Conv2dBlock(c1, 2 * self.c, 1)

        # 增强分支的 Bottleneck 序列
        self.bottlenecks = nn.ModuleList([
            Bottleneck(self.c, self.c, shortcut=shortcut, e=bottleneck_e)
            for _ in range(n)
        ])

        # cv2: 融合输出 (2 个初始分支 + n 个 Bottleneck 输出)
        self.cv2 = Conv2dBlock((2 + n) * self.c, c2, 1, act=False)
        self.act2 = nn.SiLU(inplace=True)

    def _apply_attention(self, x):
        """对增强分支应用双重注意力（通道筛选 + 空间边缘增强）"""
        dtype = x.dtype
        ch_sharp = self.ch_sharp.to(dtype)
        ch_thresh = self.ch_thresh.to(dtype)
        edge_sharp = self.edge_sharp.to(dtype)
        edge_thresh = self.edge_thresh.to(dtype)

        B, C, H, W = x.shape
        pad = self.pool_size // 2
        x_abs = x.abs()

        # 通道权重：max - avg
        max_ch = x_abs.view(B, C, -1).max(dim=-1, keepdim=True)[0].view(B, C, 1, 1)
        avg_ch = F.adaptive_avg_pool2d(x_abs, 1)
        diff_ch = max_ch - avg_ch
        ch_weight = torch.sigmoid(ch_sharp * (diff_ch - ch_thresh))

        # 空间边缘权重：max - min
        x_spatial = x_abs.mean(dim=1, keepdim=True)
        max_s = F.max_pool2d(x_spatial, self.pool_size, stride=1, padding=pad)
        min_s = -F.max_pool2d(-x_spatial, self.pool_size, stride=1, padding=pad)
        edge = max_s - min_s
        edge_weight = torch.sigmoid(edge_sharp * (edge - edge_thresh))

        out = x * ch_weight
        out = out * (1.0 + edge_weight)
        return out

    def forward(self, x):
        # 1. 投影
        x = self.cv1(x)                     # (B, 2*c, H, W)

        # 2. 分割
        a, b = x.chunk(2, dim=1)            # a: 增强分支, b: 恒等分支

        # 3. 增强分支：注意力 → 串行 Bottleneck
        a = self._apply_attention(a)
        # 初始输出包含恒等分支 b 和增强分支初始值 a（与原版 C2f 对齐）
        y = [b, a]
        x_ = a
        for m in self.bottlenecks:
            x_ = m(x_)
            y.append(x_)

        # 4. 拼接
        out = torch.cat(y, dim=1)           # (B, (2+n)*c, H, W)

        # 5. 融合输出
        out = self.cv2(out)
        out = self.act2(out)
        return out


# ================== 简单测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 模拟 P3 层输入：batch=2, 通道=128, 空间=80x80
    x = torch.randn(2, 128, 80, 80).to(device)

    # 实例化：输出 128 通道，e=0.25（与原版 C3k2 类似），n=1
    model = ChannelAwareEdgeEnhance(
        c1=128, c2=128, n=1, e=0.25, pool_size=3,
        ch_sharp=5.0, ch_thresh=0.5,
        edge_sharp=5.0, edge_thresh=0.5,
        bottleneck_e=1.0, shortcut=True
    ).to(device)

    print(model)

    y = model(x)
    print(f"Input: {x.shape} → Output: {y.shape}")

    # 形状检查
    expected_ch = 128
    expected_shape = (2, expected_ch, 80, 80)
    assert y.shape == expected_shape, f"Shape mismatch: {y.shape} vs {expected_shape}"
    print("✅ Output shape verified.")

    # 梯度测试
    loss = y.mean()
    loss.backward()
    print("✅ Backward passed.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 计算理论参数量（粗略对比原版 C3k2）
    c_ = int(128 * 0.25)  # 32
    # cv1: 128->64
    p_cv1 = 128 * 64
    # Bottleneck (1个): 32->32, 内部 32->32 无压缩
    p_bn = 32*32 + 32*32*9  # 简化
    # cv2: (2+1)*32 ->128 = 96->128
    p_cv2 = 96 * 128
    print(f"Estimated params: cv1={p_cv1}, bottleneck={p_bn}, cv2={p_cv2}, total={p_cv1+p_bn+p_cv2}")