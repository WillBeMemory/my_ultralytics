import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.DeformableConv2d import DeformableConv2d
from ultralytics.nn.modules.DepthwiseSeparableConv import DepthwiseSeparableConvWithWTConv2d

__all__ = ['TernaryDPConv']


class PruneBranch(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class TernaryDPConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 n: int = 1,
                 target_ratios: list = None,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 groups: int = 1,
                 bias: bool = False,
                 tau: float = 1.0,
                 hard: bool = False,
                 reg_weight: float = 0.01,
                 shortcut: bool = True):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.tau = tau
        self.hard = hard
        self.reg_weight = reg_weight
        self.shortcut = shortcut and (stride == 1)

        if target_ratios is None:
            target_ratios = [0.2, 0.4, 0.4]
        self.register_buffer('target_ratios', torch.tensor(target_ratios))

        # 初始化 logits（基于 in_channels）
        logits_init = torch.log(torch.tensor(target_ratios) + 1e-8)
        logits_init = self.tau * logits_init
        self.logits = nn.Parameter(logits_init.unsqueeze(0).repeat(self.in_channels, 1))

        # 三个分支
        # self.conv_branch = DeformableConv2d(
        #     in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias
        # )
        self.conv_branch = DepthwiseSeparableConvWithWTConv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        # identity 分支投影
        if in_channels != out_channels or stride != 1:
            self.identity_branch = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.identity_branch = nn.Identity()
        self.prune_branch = PruneBranch()

        # shortcut 分支
        if self.shortcut:
            if in_channels != out_channels or stride != 1:
                self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            else:
                self.shortcut_conv = nn.Identity()
        else:
            self.shortcut_conv = None

        self.register_buffer('current_probs', torch.zeros(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.in_channels, f"Input channels {C} != in_channels {self.in_channels}"

        # 计算分支选择概率
        logits = self.logits.unsqueeze(0).expand(B, -1, -1)  # (B, in_channels, 3)
        gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8))
        y = logits + gumbel
        probs = F.softmax(y / self.tau, dim=-1)

        if self.hard:
            hard_probs = torch.zeros_like(probs).scatter_(2, probs.argmax(dim=-1, keepdim=True), 1.0)
            probs = (hard_probs - probs).detach() + probs

        # 各分支输出
        out_conv = self.conv_branch(x)          # (B, out_channels, H', W')
        out_identity = self.identity_branch(x)  # (B, out_channels, H', W')
        out_prune = self.prune_branch(out_conv) # 与 out_conv 同形状

        # 将三个分支堆叠，形状 (B, 3, out_channels, H', W')
        stacked = torch.stack([out_prune, out_identity, out_conv], dim=1)

        # 全局概率：对输入通道维度取平均，得到每个样本的 3 个标量权重
        global_probs = probs.mean(dim=1)  # (B, 3)
        weights = global_probs.view(B, 3, 1, 1, 1)  # (B, 3, 1, 1, 1)

        # 加权求和
        out = (stacked * weights).sum(dim=1)  # (B, out_channels, H', W')

        if self.shortcut_conv is not None:
            out = out + self.shortcut_conv(x)

        with torch.no_grad():
            self.current_probs = probs.mean(dim=(0, 1))

        return out

    def regularization_loss(self) -> torch.Tensor:
        logits = self.logits                     # (in_channels, 3)
        probs = F.softmax(logits / self.tau, dim=-1)
        current_probs = probs.mean(dim=0)        # (3,)
        target = self.target_ratios.to(current_probs.device)
        kl = (current_probs * (torch.log(current_probs + 1e-8) - torch.log(target + 1e-8))).sum()
        return self.reg_weight * kl

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 测试配置：输入通道64，输出通道128
    in_channels = 64
    out_channels = 128
    batch_size = 2
    spatial_size = 32

    # 创建模型
    model = TernaryDPConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        tau=1.0,
        hard=False,
        target_ratios=[0.2, 0.3, 0.5],
        reg_weight=0.01,
        shortcut=True
    ).to(device)

    print(f"Model created with in_channels={in_channels}, out_channels={out_channels}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # 生成随机输入
    x = torch.randn(batch_size, in_channels, spatial_size, spatial_size).to(device)
    print(f"Input shape: {x.shape}")

    # 前向传播
    out = model(x)
    print(f"Output shape: {out.shape}")

    # 计算损失并反向传播
    loss = out.mean() + model.regularization_loss()
    loss.backward()
    print("Backward pass completed successfully.")

    # 打印分支选择概率（无梯度）
    print(f"Branch probabilities (prune, identity, conv): {model.current_probs.cpu().numpy()}")

    # 测试 stride=2 的情况（下采样）
    model_stride2 = TernaryDPConv(
        in_channels=64,
        out_channels=128,
        stride=2,
        padding=1,
        shortcut=False  # stride=2 时 shortcut 自动失效
    ).to(device)

    x2 = torch.randn(2, 64, 32, 32).to(device)
    out2 = model_stride2(x2)
    print(f"Stride=2 test - Input: {x2.shape}, Output: {out2.shape}")

    # 测试 in_channels == out_channels 的情况
    model_same = TernaryDPConv(
        in_channels=64,
        out_channels=64,
        stride=1
    ).to(device)
    x3 = torch.randn(2, 64, 32, 32).to(device)
    out3 = model_same(x3)
    print(f"Same channels test - Input: {x3.shape}, Output: {out3.shape}")

    print("All tests passed!")