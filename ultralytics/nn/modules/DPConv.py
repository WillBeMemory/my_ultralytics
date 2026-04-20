import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.DepthwiseSeparableConv import DepthwiseSeparableConvWithWTConv2d

__all__ = ['DPConv', 'DPConvBlock']

class DPConv(nn.Module):
    """
    动态部分卷积 (Dynamic Partial Convolution)，卷积分支使用深度可分离小波卷积。
    要求: in_channels == out_channels
    输入: (B, C, H, W)
    输出: (B, C, H, W)
    """
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 n=1,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 groups=1,
                 bias=False,
                 tau=1.0,
                 hard=False,
                 target_ratio=0.5,
                 reg_weight=0.001,
                 shortcut=True):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        assert in_channels == out_channels, "DPConv requires in_channels == out_channels"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.tau = tau
        self.hard = hard
        self.target_ratio = target_ratio
        self.reg_weight = reg_weight
        # 仅在 stride=1 时启用残差和动态融合
        self.use_dynamic = (stride == 1)
        self.shortcut = shortcut and self.use_dynamic

        # 可学习门控 logits
        self.logits = nn.Parameter(torch.randn(in_channels, 2) * 0.01)

        # 卷积分支：深度可分离小波卷积（保持尺寸，下采样由 stride 控制）
        # 注意：小波卷积内部会处理奇数尺寸，输出尺寸与输入相同（当 stride=1）
        self.conv = DepthwiseSeparableConvWithWTConv2d(in_channels, out_channels, k=kernel_size, s=stride)

        # 另一分支：恒等映射（仅当 stride=1 时有效）
        self.other = nn.Identity() if self.use_dynamic else None

        # 残差分支（仅当 stride=1 时有效）
        if self.shortcut:
            self.shortcut_conv = nn.Identity()
        else:
            self.shortcut_conv = None

    def forward(self, x):
        B, C, H, W = x.shape

        # Gumbel-Softmax 采样（仅在动态模式下需要）
        if self.use_dynamic:
            logits = self.logits
            gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8))
            y = logits + gumbel
            y = F.softmax(y / self.tau, dim=-1)
            if self.hard:
                y_hard = torch.zeros_like(y).scatter_(1, y.argmax(dim=-1, keepdim=True), 1.0)
                y = (y_hard - y).detach() + y
            conv_mask = y[:, 0:1].view(1, C, 1, 1)   # (1, C, 1, 1)
        else:
            # stride > 1 时退化为标准卷积（仅卷积分支）
            out = self.conv(x)
            # 记录当前比例（全卷积）
            self.current_ratio = torch.tensor(1.0, device=x.device)
            return out

        # 动态融合（stride == 1）
        out_conv = self.conv(x)
        out_other = self.other(x)
        out = conv_mask * out_conv + (1 - conv_mask) * out_other

        # 残差连接
        if self.shortcut_conv is not None:
            out = out + x

        # 记录当前 batch 的卷积比例（用于正则化）
        with torch.no_grad():
            self.current_ratio = conv_mask.mean()

        return out

    # def regularization_loss(self):
    #     """返回基于当前 batch 的比例的正则化损失（有梯度）"""
    #     diff = self.current_ratio - self.target_ratio
    #     return self.reg_weight * (diff ** 2)
    def regularization_loss(self):
        p = self.current_ratio
        q = self.target_ratio
        # KL 散度（注意：需要将标量视为伯努利分布？不适用，可以用简单的 L1 或平滑 L1）
        # 更简单：使用 Smooth L1 Loss，对小偏差不敏感
        loss = F.smooth_l1_loss(p, q)  # 或 F.l1_loss
        return self.reg_weight * loss


class DPConvBlock(nn.Module):
    """
    DPConv 模块的包装，支持输入输出通道不同。
    内部使用 1x1 卷积调整通道，然后应用 DPConv。
    """
    def __init__(self, in_channels, out_channels,n=1, **dpconv_kwargs):
        super().__init__()
        # 1x1 卷积用于调整通道数
        self.proj = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
        # DPConv 模块（输入输出通道均为 out_channels）
        self.dpconv = DPConv(out_channels, out_channels, **dpconv_kwargs)

    def forward(self, x):
        x = self.proj(x)
        return self.dpconv(x)


# 测试代码
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 测试 DPConv（通道相同，stride=1）
    print("\n=== Test DPConv (in=out=64, stride=1) ===")
    model = DPConv(64, 64, kernel_size=3, stride=1, tau=1.0, hard=False,
                   target_ratio=0.5, reg_weight=0.01, shortcut=True).to(device)
    x = torch.randn(2, 64, 64, 64).to(device)
    out = model(x)
    print(f"Output shape: {out.shape} (expected [2,64,64,64])")
    loss = out.mean() + model.regularization_loss()
    loss.backward()
    print("Backward OK")

    # 测试 DPConv（通道相同，stride=2，应退化为标准小波卷积）
    print("\n=== Test DPConv (in=out=64, stride=2) ===")
    model2 = DPConv(64, 64, kernel_size=3, stride=2, tau=1.0, hard=False,
                    target_ratio=0.5, reg_weight=0.01, shortcut=True).to(device)
    x2 = torch.randn(2, 64, 64, 64).to(device)
    out2 = model2(x2)
    print(f"Output shape: {out2.shape} (expected [2,64,32,32])")
    loss2 = out2.mean()
    # stride>1 时没有动态选择，regularization_loss 返回基于固定比例1.0的损失
    loss2 = loss2 + model2.regularization_loss()
    loss2.backward()
    print("Backward OK")

    # 测试 DPConvBlock（通道不同）
    print("\n=== Test DPConvBlock (64->128, stride=1) ===")
    block = DPConvBlock(64, 128, kernel_size=3, stride=1, tau=1.0, hard=False,
                        target_ratio=0.5, reg_weight=0.01, shortcut=True).to(device)
    x3 = torch.randn(2, 64, 64, 64).to(device)
    out3 = block(x3)
    print(f"Output shape: {out3.shape} (expected [2,128,64,64])")
    loss3 = out3.mean() + block.dpconv.regularization_loss()
    loss3.backward()
    print("Backward OK")

    print("\nAll tests passed!")