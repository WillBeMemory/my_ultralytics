import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAwareEdgeEnhance(nn.Module):
    """
    通道对比度筛选 + 空间边缘增强模块 (符合 YOLO 参数习惯)

    Args:
        c1         : 输入通道数
        c2         : 输出通道数（通常与 c1 一致）
        n          : 占位参数（兼容 YOLO 配置风格，实际不执行重复）
        pool_size  : 池化窗口大小（默认 3）
        ch_sharp   : 通道权重 sigmoid 陡峭度 (α)
        ch_thresh  : 通道权重 sigmoid 偏移 (τ)
        edge_sharp : 边缘权重 sigmoid 陡峭度 (β)
        edge_thresh: 边缘权重 sigmoid 偏移 (γ)
        use_conv   : 是否在末尾添加 1x1 卷积进行通道混合
    """
    def __init__(self, c1, c2, n=1, pool_size=3, ch_sharp=5.0, ch_thresh=0.5,
                 edge_sharp=5.0, edge_thresh=0.5, use_conv=True):
        super().__init__()
        # 输入输出通道处理（兼容 c2 != c1 的情况）
        self.c1 = c1
        self.c2 = c2
        self.n = n          # 仅占位，不影响逻辑
        self.pool_size = pool_size
        self.ch_sharp = nn.Parameter(torch.tensor(ch_sharp), requires_grad=False)
        self.ch_thresh = nn.Parameter(torch.tensor(ch_thresh), requires_grad=False)
        self.edge_sharp = nn.Parameter(torch.tensor(edge_sharp), requires_grad=False)
        self.edge_thresh = nn.Parameter(torch.tensor(edge_thresh), requires_grad=False)

        # 可选的输出卷积（若 c2 != c1 则强制使用）
        if use_conv or c1 != c2:
            self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        else:
            self.conv = nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        pad = self.pool_size // 2

        # 1. 通道对比度筛选（基于绝对值）
        x_abs = x.abs()
        max_ch = F.adaptive_max_pool2d(x_abs, 1)  # (B, C, 1, 1)
        avg_ch = F.adaptive_avg_pool2d(x_abs, 1)  # (B, C, 1, 1)
        diff_ch = max_ch - avg_ch
        ch_weight = torch.sigmoid(self.ch_sharp * (diff_ch - self.ch_thresh))  # (B, C, 1, 1)

        # 2. 空间边缘增强（共享权重，基于 max - min）
        x_spatial = x_abs.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        max_s = F.max_pool2d(x_spatial, self.pool_size, stride=1, padding=pad)
        min_s = -F.max_pool2d(-x_spatial, self.pool_size, stride=1, padding=pad)
        edge = max_s - min_s
        edge_weight = torch.sigmoid(self.edge_sharp * (edge - self.edge_thresh))  # (B, 1, H, W)

        # 3. 应用增强
        out = x * ch_weight                # 通道抑制
        out = out * (1.0 + edge_weight)    # 边缘残差增强

        # 4. 输出变换（通道数调整）
        out = self.conv(out)
        return out


if __name__ == "__main__":
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 参数配置
    batch_size = 2
    c1 = 256  # 输入通道
    c2 = 256  # 输出通道
    H, W = 20, 20  # 典型 P5 特征图尺寸

    # 创建随机输入
    x = torch.randn(batch_size, c1, H, W).to(device)

    # 实例化模块（n 仅占位）
    model = ChannelAwareEdgeEnhance(c1, c2, n=1, pool_size=3).to(device)
    print(model)

    # 前向传播
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")

    # 检查通道数
    assert y.shape[1] == c2, f"Output channels {y.shape[1]} != {c2}"

    # 简单反向传播测试
    loss = y.mean()
    loss.backward()
    print("Backward pass completed successfully.")

    # 检查梯度是否存在于输入
    if x.grad is None:
        print("Warning: Input gradient is None (expected if x requires_grad=False)")
    else:
        print(f"Input gradient shape: {x.grad.shape}")

    print("All tests passed!")

