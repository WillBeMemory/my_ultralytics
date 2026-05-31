import torch
import torch.nn as nn
from ultralytics.nn.modules.block import Conv


class StarBlock(nn.Module):
    """StarBlock: 双分支变换后逐元素相乘 + 残差连接"""
    def __init__(self, c, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c * e)  # 隐藏通道数
        self.shortcut = shortcut

        # 两个平行的 1x1 卷积，用于生成不同特征空间的投影
        self.conv1 = Conv(c, c_, 1, 1, g=g, act=False)
        self.conv2 = Conv(c, c_, 1, 1, g=g, act=False)

        # 将相乘后的结果投影回原始通道
        self.fusion = Conv(c_, c, 1, 1, g=g, act=False)

        # 可选的恒等连接
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)  # 分支1
        x2 = self.conv2(x)  # 分支2
        # 星操作核心：逐元素相乘（类比注意力机制中的 Query * Key）
        star = x1 * x2
        # 融合投影
        out = self.fusion(star)

        if self.shortcut:
            out = self.act(out + identity)
        else:
            out = self.act(out)
        return out


class StarC3k2(nn.Module):
    """
    基于星操作的特征精炼层，兼容 C3k2 参数接口。

    参数:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        n (int): 星操作块的重复次数
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
        self.n = n

        # CSP 标准结构：cv1 投影到 2*c
        self.cv1 = Conv(c1, 2 * self.c, 1, 1, g=g)

        # 增强分支：n 个 StarBlock 串联
        self.m = nn.ModuleList([
            StarBlock(self.c, shortcut=shortcut, g=g, e=1.0) for _ in range(n)
        ])

        # cv2 融合：将直通分支 + 初始增强分支 + 每个 StarBlock 的输出合并
        self.cv2 = Conv((2 + n) * self.c, c2, 1, g=g, act=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # 1. 通道分割
        y = list(self.cv1(x).chunk(2, dim=1))   # [b, a]
        a = y[1]   # 增强分支

        # 2. 星操作序列，保留每个块的输出（类似 C2f）
        outputs = [y[0], a]   # 直通分支 + 初始增强分支
        for m in self.m:
            a = m(a)
            outputs.append(a)

        # 3. 拼接并融合
        out = torch.cat(outputs, dim=1)
        out = self.act(self.cv2(out))
        return out


# ================== 简单测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 模拟 P4 检测头前的输入：batch=2, 通道=512, 空间=40x40
    x = torch.randn(2, 512, 40, 40).to(device)

    # 实例化 StarC3k2：输出512通道，n=1，e=0.5
    model = StarC3k2(c1=512, c2=512, n=1, e=0.5, shortcut=True).to(device)

    print(model)
    y = model(x)

    # 形状检查
    expected_shape = (2, 512, 40, 40)
    status = "✅" if y.shape == expected_shape else "❌"
    print(f"Input: {x.shape} → Output: {y.shape} expected {expected_shape} {status}")

    # 梯度测试
    loss = y.mean()
    loss.backward()
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"Backward passed. All params have gradients: {has_grad}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")