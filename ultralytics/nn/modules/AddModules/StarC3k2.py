import torch
import torch.nn as nn
from ultralytics.nn.modules.block import Conv


class StarBlock(nn.Module):
    """StarBlock: 双分支变换后逐元素相乘 + 残差连接"""
    def __init__(self, c, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c * e)
        self.shortcut = shortcut
        self.conv1 = Conv(c, c_, 1, 1, g=g, act=False)
        self.conv2 = Conv(c, c_, 1, 1, g=g, act=False)
        self.fusion = Conv(c_, c, 1, 1, g=g, act=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        star = x1 * x2
        out = self.fusion(star)
        if self.shortcut:
            out = self.act(out + identity)
        else:
            out = self.act(out)
        return out


class StarC3k2(nn.Module):
    """
    混合星操作精炼层，兼容 C3k2 参数接口。

    在 C3k2 划分出的精炼通道上，再次对半划分：
    - 一半经过 n 个 StarBlock 序列（原始逻辑）
    - 另一半经过单个 StarBlock 增强后直接与原始直通分支合并

    参数:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        n (int): StarBlock 重复次数
        c3k (bool): 占位参数，仅用于兼容 C3k2 接口
        e (float): CSP 隐藏通道扩展比，默认为 0.5
        attn (bool): 占位参数，仅用于兼容 C3k2 接口
        g (int): 分组卷积的组数，默认为 1
        shortcut (bool): StarBlock 内部是否使用残差连接
        **kwargs: 忽略其他不支持的参数
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True, **kwargs):
        super().__init__()
        self.c = int(c2 * e)          # 分支通道数
        self.c_half = self.c // 2     # 精炼分支再次划分的一半
        self.n = n

        # CSP 标准结构：cv1 投影到 2*c
        self.cv1 = Conv(c1, 2 * self.c, 1, 1, g=g)

        # 增强分支：n 个 StarBlock 串联（作用于 a1）
        self.m = nn.ModuleList([
            StarBlock(self.c_half, shortcut=shortcut, g=g, e=1.0) for _ in range(n)
        ])

        # 作用于另一半精炼通道的单次星操作增强块
        self.star_enhance = StarBlock(self.c_half, shortcut=shortcut, g=g, e=1.0)

        # cv2 融合：直通分支 + 原始精炼分支的一半 + 星操作增强后的另一半 + n 个 StarBlock 输出
        # 总拼接通道数：b(c) + a1(c_half) + a2(c_half) + n * (c_half) = c + (2+n)*c_half
        self.cv2 = Conv(self.c + (2 + n) * self.c_half, c2, 1, g=g, act=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # 1. 通道分割
        y = list(self.cv1(x).chunk(2, dim=1))   # [b, a]，b 直通，a 精炼
        b, a = y[0], y[1]

        # 2. 精炼分支再次一分为二：a1 走 StarBlock 序列，a2 走单次星操作增强
        a1, a2 = a.chunk(2, dim=1)              # 各占 self.c_half 通道

        # 3. a1 执行 StarBlock 序列（原始 C3k2 逻辑）
        outputs = [b, a1]                       # 直通分支 + 原始精炼分支的一半
        a1_out = a1
        for m in self.m:
            a1_out = m(a1_out)
            outputs.append(a1_out)              # 每个 StarBlock 的输出

        # 4. a2 执行单次星操作增强
        a2_out = self.star_enhance(a2)
        outputs.append(a2_out)

        # 5. 拼接并融合
        out = torch.cat(outputs, dim=1)
        out = self.act(self.cv2(out))
        return out


# ================== 简单测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    x = torch.randn(2, 512, 40, 40).to(device)
    model = StarC3k2(c1=512, c2=512, n=1, e=0.5, shortcut=True).to(device)

    print(model)
    y = model(x)

    expected_shape = (2, 512, 40, 40)
    status = "✅" if y.shape == expected_shape else "❌"
    print(f"Input: {x.shape} → Output: {y.shape} expected {expected_shape} {status}")

    loss = y.mean()
    loss.backward()
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"Backward passed. All params have gradients: {has_grad}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")