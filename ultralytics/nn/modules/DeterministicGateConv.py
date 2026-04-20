import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.DepthwiseSeparableConv import DepthwiseSeparableConvWithWTConv2d

__all__ = ['DeterministicGateConv']

class PruneBranch(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)

class DeterministicGateConv(nn.Module):
    """
    确定性门控卷积模块（无Gumbel噪声）
    三分支：prune(全零), identity, conv(深度可分离小波卷积)
    门控权重：每个通道独立的可学习logits，通过softmax或hard argmax得到确定性的加权系数。
    """
    def __init__(self,
                 c1,                     # 输入通道
                 c2,                     # 输出通道
                 n=1,                    # 重复次数（兼容YOLO，未使用）
                 target_ratios=None,    # 目标分支比例 [prune, identity, conv]
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 groups=1,
                 tau=5.0,               # 温度系数（softmax温度）
                 hard=False,            # 是否硬化（argmax one-hot）
                 reg_weight=0.01,       # KL散度正则化权重
                 shortcut=True):        # 是否使用残差连接（仅当stride=1时有效）
        super().__init__()
        assert c1 == c2, "DeterministicGateConv requires c1 == c2"
        self.c1 = c1
        self.c2 = c2
        self.n = n
        self.stride = stride
        self.tau = tau
        self.hard = hard
        self.reg_weight = reg_weight
        self.shortcut = shortcut and (stride == 1)

        if target_ratios is None:
            target_ratios = [0.2, 0.4, 0.4]  # prune, identity, conv
        self.register_buffer('target_ratios', torch.tensor(target_ratios))

        # 可学习的logits：每个通道有3个值，对应三个分支的未归一化权重
        # 初始化使得初始softmax输出接近目标比例
        logits_init = torch.log(torch.tensor(target_ratios) + 1e-8)  # (3,)
        logits_init = self.tau * logits_init                         # 考虑温度
        self.logits = nn.Parameter(logits_init.unsqueeze(0).repeat(self.c1, 1))  # (C,3)

        # 三个分支
        self.prune_branch = PruneBranch()
        self.identity_branch = nn.Identity()
        self.conv_branch = DepthwiseSeparableConvWithWTConv2d(
            c1, c2, k=kernel_size, s=stride, p=padding, g=groups
        )

        self.register_buffer('current_probs', torch.zeros(3))

    def forward(self, x):
        B, C, H, W = x.shape
        # logits: (C,3) -> (B,C,3)
        logits = self.logits.unsqueeze(0).expand(B, -1, -1)

        # 计算概率（确定性 softmax，无噪声）
        probs = F.softmax(logits / self.tau, dim=-1)   # (B,C,3)

        if self.hard:
            # 硬化：取 argmax 得到 one-hot，并使用直通估计器保持可微
            hard_probs = torch.zeros_like(probs).scatter_(
                2, probs.argmax(dim=-1, keepdim=True), 1.0
            )
            probs = (hard_probs - probs).detach() + probs

        # 计算三个分支输出
        out_prune = self.prune_branch(x)
        out_identity = x
        out_conv = self.conv_branch(x)

        # 如果 stride>1，conv_branch 已经下采样，需要对其他分支下采样
        if self.stride > 1:
            out_prune = F.avg_pool2d(out_prune, kernel_size=self.stride, stride=self.stride)
            out_identity = F.avg_pool2d(out_identity, kernel_size=self.stride, stride=self.stride)

        # 加权融合：probs_exp 形状 (B,C,3,1,1)
        probs_exp = probs.unsqueeze(-1).unsqueeze(-1)
        out = (probs_exp[:, :, 0] * out_prune +
               probs_exp[:, :, 1] * out_identity +
               probs_exp[:, :, 2] * out_conv)

        if self.shortcut:
            out = out + x   # 注意：此时 x 尺寸需与 out 一致，stride=1 时相同

        with torch.no_grad():
            self.current_probs = probs.mean(dim=(0, 1))   # 记录当前batch的平均分支概率

        return out

    def regularization_loss(self):
        """
        计算 KL 散度损失，约束当前平均分支比例接近 target_ratios
        """
        probs = F.softmax(self.logits / self.tau, dim=-1)
        current_probs = probs.mean(dim=0)               # (3,)
        target = self.target_ratios.to(current_probs.device)
        kl = (current_probs * (torch.log(current_probs + 1e-8) - torch.log(target + 1e-8))).sum()
        return self.reg_weight * kl