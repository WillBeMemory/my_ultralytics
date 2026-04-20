import torch
import torch.nn as nn


_all_ = ["CSP_PConv"]
# ========== CSP-PConv 模块定义 ==========
class PartialConv(nn.Module):
    """
    部分卷积模块
    Args:
        dim: 输入通道数
        n_div: 划分比例，默认4，表示取1/4的通道进行卷积
        kernel_size: 卷积核大小
    """

    def __init__(self, dim, n_div=4, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.n_div = n_div
        self.kernel_size = kernel_size
        self.partial_dim = dim // n_div  # 参与卷积的通道数

        # 仅对部分通道进行卷积
        self.conv = nn.Conv2d(
            self.partial_dim,
            self.partial_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

    def forward(self, x):
        # 拆分通道
        x_partial = x[:, :self.partial_dim, :, :]  # 前1/4通道进行卷积
        x_identity = x[:, self.partial_dim:, :, :]  # 后3/4通道直接复制

        # 对部分通道进行卷积
        x_partial = self.conv(x_partial)

        # 拼接输出
        return torch.cat([x_partial, x_identity], dim=1)


class CSP_PConv(nn.Module):
    """
    CSP-PConv 模块
    将输入分为两条路径，一条经过PConv，一条直接传递，最后融合
    """

    def __init__(self, in_channels, out_channels, n=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        hidden_channels = out_channels // 2

        # 第一条路径：先1x1降维，再PConv
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.pconv = PartialConv(hidden_channels)

        # 第二条路径：直接1x1降维
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)

        # 最终融合卷积
        self.final_conv = nn.Conv2d(hidden_channels * 2, out_channels, 1, bias=False)
        self.final_bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # 分支1：PConv路径
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.pconv(x1)

        # 分支2：直接传递路径
        x2 = self.conv2(x)
        x2 = self.bn2(x2)

        # 拼接两条路径
        out = torch.cat([x1, x2], dim=1)
        out = self.final_conv(out)
        out = self.final_bn(out)

        # 残差连接
        if self.shortcut and x.shape == out.shape:
            out = out + x

        return self.act(out)


# ========== 测试代码 ==========
def test_csp_pconv():
    print("=" * 50)
    print("测试 CSP-PConv 模块")
    print("=" * 50)

    # 设置随机种子
    torch.manual_seed(42)

    # 测试用例 1: 输入输出通道相同，shortcut=True
    print("\n[测试1] in_channels=64, out_channels=64, shortcut=True")
    batch = 2
    h, w = 32, 32
    in_c, out_c = 64, 64

    x = torch.randn(batch, in_c, h, w).requires_grad_(True)
    model = CSP_PConv(in_c, out_c, shortcut=True)

    out = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    assert out.shape == (batch, out_c, h, w), f"输出形状错误: {out.shape}"

    # 测试梯度
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "输入梯度为 None"
    assert model.conv1.weight.grad is not None, "conv1 梯度为 None"
    assert model.pconv.conv.weight.grad is not None, "pconv 梯度为 None"
    print("✓ 梯度反向传播正常")

    # 测试用例 2: 输入输出通道不同，shortcut=False
    print("\n[测试2] in_channels=128, out_channels=256, shortcut=False")
    in_c, out_c = 128, 256
    x = torch.randn(batch, in_c, h, w).requires_grad_(True)
    model = CSP_PConv(in_c, out_c, shortcut=False)

    out = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    assert out.shape == (batch, out_c, h, w), f"输出形状错误: {out.shape}"

    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "输入梯度为 None"
    print("✓ 梯度反向传播正常")

    # 测试用例 3: 多尺度输入 (不同空间尺寸)
    print("\n[测试3] 输入尺寸 64x64")
    h, w = 64, 64
    x = torch.randn(batch, 64, h, w).requires_grad_(True)
    model = CSP_PConv(64, 128)

    out = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    assert out.shape == (batch, 128, h, w), f"输出形状错误: {out.shape}"
    print("✓ 输出形状正确")

    # 测试用例 4: 检查 PConv 内部通道拆分逻辑
    print("\n[测试4] 验证 PConv 通道拆分")
    dim = 32
    pconv = PartialConv(dim, n_div=4)
    x_p = torch.randn(1, dim, 8, 8)
    out_p = pconv(x_p)
    print(f"PConv 输入通道: {dim}, 输出通道: {out_p.shape[1]}")
    assert out_p.shape[1] == dim, f"PConv 输出通道数应为 {dim}, 实际 {out_p.shape[1]}"
    print("✓ PConv 通道保持正确")

    print("\n" + "=" * 50)
    print("所有测试通过！CSP-PConv 模块工作正常。")
    print("=" * 50)


if __name__ == "__main__":
    test_csp_pconv()