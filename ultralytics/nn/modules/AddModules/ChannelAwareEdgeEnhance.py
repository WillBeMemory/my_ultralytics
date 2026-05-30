import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv, Bottleneck


class ChannelAwareEdgeEnhance(nn.Module):
    """
    多通道边缘增强模块（基于 C3k2 结构）

    在增强分支进入 Bottleneck 前，对每个通道独立进行局部极差
    (max - min) 边缘增强，强化舰船轮廓等细粒度特征。

    YAML 参数示例：
    [-1, 1, ChannelAwareEdgeEnhance, [128, 1, 0.5, 3, 5.0, 0.5, True]]
    参数说明：c2, n, e, pool_size, edge_sharp, edge_thresh, shortcut
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        e: float = 0.5,
        pool_size: int = 3,
        edge_sharp: float = 5.0,
        edge_thresh: float = 0.5,
        shortcut: bool = True,
        g: int = 1,
        **kwargs
    ):
        super().__init__()
        self.c = int(c2 * e)          # 分支通道数
        self.n = n
        self.pool_size = pool_size

        self.register_buffer('edge_sharp', torch.tensor(edge_sharp))
        self.register_buffer('edge_thresh', torch.tensor(edge_thresh))

        # CSP 1x1 投影
        self.cv1 = Conv(c1, 2 * self.c, 1)

        # 增强分支的 Bottleneck 序列
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

        # 拼接融合层
        self.cv2 = Conv((2 + n) * self.c, c2, 1, act=False)
        self.act2 = nn.SiLU(inplace=True)

    def _apply_edge_enhance(self, x: torch.Tensor) -> torch.Tensor:
        """逐通道局部极差边缘增强"""
        B, C, H, W = x.shape
        dtype = x.dtype
        edge_sharp = self.edge_sharp.to(dtype)
        edge_thresh = self.edge_thresh.to(dtype)

        # 将通道维度并入 batch，便于逐通道处理
        x_flat = x.reshape(B * C, 1, H, W)

        pad = self.pool_size // 2
        max_s = F.max_pool2d(x_flat, self.pool_size, stride=1, padding=pad)
        min_s = -F.max_pool2d(-x_flat, self.pool_size, stride=1, padding=pad)
        edge = max_s - min_s   # (B*C, 1, H, W)

        # 生成权重并增强
        edge_weight = torch.sigmoid(edge_sharp * (edge - edge_thresh))  # (B*C, 1, H, W)
        edge_weight = edge_weight.view(B, C, H, W)

        return x * (1.0 + edge_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))   # [b, a]
        a = y[1]   # 增强分支
        a = self._apply_edge_enhance(a)
        y = [y[0], a]   # 恒等分支 b + 增强后的 a
        x_ = a
        for m in self.m:
            x_ = m(x_)
            y.append(x_)

        out = torch.cat(y, dim=1)
        out = self.cv2(out)
        return self.act2(out)


# ================== 简单测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    x = torch.randn(2, 128, 80, 80).to(device)

    model = ChannelAwareEdgeEnhance(
        c1=128, c2=128, n=1, e=0.5, pool_size=3,
        edge_sharp=5.0, edge_thresh=0.5, shortcut=True
    ).to(device)

    y = model(x)
    print(f"Input: {x.shape} → Output: {y.shape}")

    loss = y.mean()
    loss.backward()
    print("✅ Forward & backward passed.")