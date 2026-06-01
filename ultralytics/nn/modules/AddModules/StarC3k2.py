import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv


class StarBlock(nn.Module):
    """稳定版 StarBlock：双分支变换后逐元素相乘 + 残差连接，含数值稳定机制"""
    def __init__(self, c, shortcut=True, g=1, e=0.5, temperature_init=0.1):
        super().__init__()
        c_ = int(c * e)
        self.shortcut = shortcut

        self.conv1 = Conv(c, c_, 1, 1, g=g, act=False)
        self.conv2 = Conv(c, c_, 1, 1, g=g, act=False)
        self.fusion = Conv(c_, c, 1, 1, g=g, act=False)

        # 可学习的温度系数，初始化为较小值，防止初期爆炸
        self.temperature = nn.Parameter(torch.tensor(temperature_init))

        # 可选：在乘法之后添加 BatchNorm 进一步稳定分布
        self.star_bn = nn.BatchNorm2d(c_)

        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)          # (B, c_, H, W)
        x2 = self.conv2(x)

        # 数值稳定关键：对两个投影向量做 L2 归一化，限制值域在 [-1, 1] 附近
        x1 = F.normalize(x1, p=2, dim=1, eps=1e-6)
        x2 = F.normalize(x2, p=2, dim=1, eps=1e-6)

        # 逐元素相乘，受温度系数控制
        star = x1 * x2 * self.temperature

        # 通过 BN 稳定分布
        star = self.star_bn(star)

        out = self.fusion(star)

        if self.shortcut:
            out = self.act(out + identity)
        else:
            out = self.act(out)
        return out


class StarC3k2(nn.Module):
    """
    稳定版星操作特征精炼层，兼容 C3k2 参数接口。

    参数:
        c1, c2, n, c3k, e, attn, g, shortcut: 同 C3k2
        temperature_init (float): StarBlock 初始温度系数，建议 0.05~0.1
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1,
                 shortcut=True, temperature_init=0.1, **kwargs):
        super().__init__()
        self.c = int(c2 * e)
        self.n = n

        self.cv1 = Conv(c1, 2 * self.c, 1, 1, g=g)
        self.m = nn.ModuleList([
            StarBlock(self.c, shortcut=shortcut, g=g, e=1.0,
                      temperature_init=temperature_init) for _ in range(n)
        ])
        self.cv2 = Conv((2 + n) * self.c, c2, 1, g=g, act=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))   # [b, a]
        a = y[1]
        outputs = [y[0], a]
        for m in self.m:
            a = m(a)
            outputs.append(a)
        out = torch.cat(outputs, dim=1)
        out = self.act(self.cv2(out))
        return out


# 测试（略）

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