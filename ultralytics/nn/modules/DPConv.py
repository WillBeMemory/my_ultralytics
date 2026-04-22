# ============================================
# File: ultralytics/nn/modules/DPConv.py
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.DepthwiseSeparableConv import DepthwiseSeparableConvWithWTConv2d

__all__ = ['DPConv']


class DPConv(nn.Module):
    """
    动态部分卷积 (C3风格)，支持通道压缩、动态门控、小波卷积，并处理 stride>1 时的下采样对齐。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        e: float = 0.5,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        bias: bool = False,
        tau: float = 1.0,
        hard: bool = False,
        target_ratio: float = 0.5,
        reg_weight: float = 0.001,
        shortcut: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.e = e
        self.stride = stride
        self.tau = tau
        self.hard = hard
        self.target_ratio = target_ratio
        self.reg_weight = reg_weight
        self.shortcut = shortcut and (stride == 1)  # 外部残差仅当 stride=1 且通道匹配时有效

        c_ = int(out_channels * e)

        # 主路径压缩 (cv1)
        self.cv1 = nn.Conv2d(in_channels, c_, kernel_size=1, bias=False)
        # 捷径分支压缩 (cv2)
        self.cv2 = nn.Conv2d(in_channels, c_, kernel_size=1, bias=False)

        # 处理 stride > 1 时捷径分支的下采样
        if stride > 1:
            # 使用平均池化进行下采样，与 YOLO 风格一致
            self.cv2_downsample = nn.AvgPool2d(stride, stride)
        else:
            self.cv2_downsample = nn.Identity()

        # 动态门控（仅 stride=1 时启用）
        self.use_dynamic = (stride == 1)
        if self.use_dynamic:
            self.logits = nn.Parameter(torch.randn(c_, 2) * 0.01)

        # 主路径的卷积操作：深度可分离小波卷积（内部已处理 stride）
        self.conv = DepthwiseSeparableConvWithWTConv2d(
            c_, c_, k=kernel_size, s=stride, p=padding, g=groups
        )

        self.other = nn.Identity() if self.use_dynamic else None

        # 输出融合：拼接后 1x1 升维
        self.cv3 = nn.Conv2d(2 * c_, out_channels, kernel_size=1, bias=False)

        # 外部残差连接（仅当 stride=1 且通道匹配时启用）
        if self.shortcut and in_channels == out_channels:
            self.shortcut_conv = nn.Identity()
        elif self.shortcut:
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut_conv = None

        self.register_buffer('current_ratio', torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 双路 1x1 压缩
        x1 = self.cv1(x)                     # 主路径入口
        x2 = self.cv2(x)                     # 捷径分支入口
        x2 = self.cv2_downsample(x2)         # 对齐 stride > 1 时的尺寸

        if self.use_dynamic:
            # 动态门控选择
            logits = self.logits
            gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8))
            y = logits + gumbel
            y = F.softmax(y / self.tau, dim=-1)
            if self.hard:
                y_hard = torch.zeros_like(y).scatter_(1, y.argmax(dim=-1, keepdim=True), 1.0)
                y = (y_hard - y).detach() + y
            conv_mask = y[:, 0:1].view(1, -1, 1, 1)

            out_conv = self.conv(x1)
            out_other = self.other(x1)
            x1_processed = conv_mask * out_conv + (1 - conv_mask) * out_other

            with torch.no_grad():
                self.current_ratio = conv_mask.mean()
        else:
            # stride > 1：退化为纯小波卷积
            x1_processed = self.conv(x1)
            with torch.no_grad():
                self.current_ratio = torch.tensor(1.0, device=x.device)

        # 主路径与捷径拼接
        out = torch.cat([x1_processed, x2], dim=1)

        # 1x1 卷积恢复通道
        out = self.cv3(out)

        # 外部残差连接（仅 stride=1 时有效）
        if self.shortcut_conv is not None:
            out = out + self.shortcut_conv(x)

        return out

    def regularization_loss(self) -> torch.Tensor:
        p = self.current_ratio
        q = torch.tensor(self.target_ratio, device=p.device)
        loss = F.smooth_l1_loss(p, q)
        return self.reg_weight * loss


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 测试1：stride=1 动态模式
    print("\n=== Test DPConv (stride=1, e=0.5) ===")
    model = DPConv(64, 128, e=0.5, stride=1).to(device)
    x = torch.randn(2, 64, 64, 64).to(device)
    out = model(x)
    print(f"Output shape: {out.shape} (expected [2, 128, 64, 64])")
    loss = out.mean() + model.regularization_loss()
    loss.backward()
    print("Backward OK")

    # 测试2：stride=2 退化模式（已修复尺寸不匹配）
    print("\n=== Test DPConv (stride=2, fallback) ===")
    model2 = DPConv(64, 128, e=0.5, stride=2, shortcut=False).to(device)
    x2 = torch.randn(2, 64, 64, 64).to(device)
    out2 = model2(x2)
    print(f"Output shape: {out2.shape} (expected [2, 128, 32, 32])")
    loss2 = out2.mean() + model2.regularization_loss()
    loss2.backward()
    print("Backward OK")
    print(f"Conv branch ratio: {model2.current_ratio.item():.3f} (should be 1.0)")

    # 测试3：in=out=64, stride=1, shortcut=True
    print("\n=== Test DPConv (in=out=64, shortcut) ===")
    model3 = DPConv(64, 64, e=0.5, stride=1, shortcut=True).to(device)
    x3 = torch.randn(2, 64, 64, 64).to(device)
    out3 = model3(x3)
    print(f"Output shape: {out3.shape} (expected [2, 64, 64, 64])")
    loss3 = out3.mean() + model3.regularization_loss()
    loss3.backward()
    print("Backward OK")

    print("\nAll tests passed!")