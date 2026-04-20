import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.SCSA import SCSA
from ultralytics.nn.modules.DepthwiseSeparableConv import DepthwiseSeparableConvWithWTConv2d

# from .SCSA import SCSA
# from .DepthwiseSeparableConv import DepthwiseSeparableConvWithWTConv2d

__all = ["DynamicPartialIdentity","DynamicWaveletAttentionIdentity","DynamicWaveletIdentity"]
class PartialConv2d(nn.Module):
    """对一部分通道进行3x3卷积，其余恒等（输入输出通道数相同）"""
    def __init__(self, channels, split_ratio=0.5):
        super().__init__()
        self.split_ratio = split_ratio
        self.c_conv = int(channels * split_ratio)
        self.c_id = channels - self.c_conv
        if self.c_conv > 0:
            self.conv = nn.Conv2d(self.c_conv, self.c_conv, 3, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(self.c_conv)
            self.act = nn.SiLU(inplace=True)
        else:
            self.conv = nn.Identity()
            self.bn = nn.Identity()
            self.act = nn.Identity()

    def forward(self, x):
        if self.c_conv == 0:
            return x
        if self.c_conv == x.shape[1]:
            out = self.conv(x)
            out = self.bn(out)
            out = self.act(out)
            return out
        x_conv, x_id = x[:, :self.c_conv], x[:, self.c_conv:]
        out_conv = self.conv(x_conv)
        out_conv = self.bn(out_conv)
        out_conv = self.act(out_conv)
        return torch.cat([out_conv, x_id], dim=1)


class DynamicMultiBranch(nn.Module):
    """多分支动态选择模块（全分支计算，加权融合）"""
    def __init__(self, in_channels, branches, tau=1.0, hard=False,
                 target_ratios=None, reg_weight=0.01, entropy_weight=0.01):
        super().__init__()
        self.num_branches = len(branches)
        self.branches = nn.ModuleList(branches)
        self.tau = tau
        self.hard = hard
        self.target_ratios = target_ratios
        self.reg_weight = reg_weight
        self.entropy_weight = entropy_weight
        self.logits = nn.Parameter(torch.randn(in_channels, self.num_branches) * 0.01)
        self.register_buffer('avg_probs', torch.zeros(self.num_branches))

    def forward(self, x):
        B, C, H, W = x.shape
        logits = self.logits.unsqueeze(0).expand(B, -1, -1)
        gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8))
        y = logits + gumbel
        probs = F.softmax(y / self.tau, dim=-1)  # (B, C, K)

        if self.hard:
            hard_probs = torch.zeros_like(probs).scatter_(2, probs.argmax(dim=-1, keepdim=True), 1.0)
            probs = (hard_probs - probs).detach() + probs

        # 计算所有分支的输出
        branch_outputs = [branch(x) for branch in self.branches]

        # 加权融合
        probs_expanded = probs.unsqueeze(-1).unsqueeze(-1)  # (B, C, K, 1, 1)
        out = sum(probs_expanded[:, :, k] * out_b for k, out_b in enumerate(branch_outputs))

        with torch.no_grad():
            self.avg_probs = probs.mean(dim=(0, 1))  # (K,)

        return out

    def regularization_loss(self):
        """目标比例正则化 + 熵正则化"""
        loss = 0.0
        if self.target_ratios is not None:
            target = torch.tensor(self.target_ratios, device=self.avg_probs.device)
            loss += self.reg_weight * F.mse_loss(self.avg_probs, target)
        # 熵正则化：鼓励分布尖锐，减少分支活跃数
        if self.entropy_weight > 0:
            # 计算每个通道的熵，然后取平均
            entropy = - (self.avg_probs * torch.log(self.avg_probs + 1e-8)).sum()
            loss += self.entropy_weight * entropy
        return loss


class DynamicPartialIdentity(nn.Module):
    """可替换 YOLO 中 C3k2 的动态块"""
    def __init__(self, c1, c2, n=1, split_ratio=0.5, tau=1.0, hard=False,
                 target_ratios=None, reg_weight=0.01, entropy_weight=0.01, shortcut=True):
        super().__init__()
        self.c = c2
        self.n = n
        self.shortcut = shortcut and c1 == c2

        layers = []
        for _ in range(n):
            branch_partial = PartialConv2d(c1, split_ratio=split_ratio)
            branch_identity = nn.Identity()
            branches = [branch_partial, branch_identity]
            dynamic_block = DynamicMultiBranch(
                in_channels=c1,
                branches=branches,
                tau=tau,
                hard=hard,
                target_ratios=target_ratios or [0.5, 0.5],
                reg_weight=reg_weight,
                entropy_weight=entropy_weight
            )
            layers.append(dynamic_block)

        self.dynamic_seq = nn.Sequential(*layers)

        # 通道调整
        self.conv = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        out = self.dynamic_seq(x)
        if self.shortcut:
            out = out + x
        return self.conv(out)

    def regularization_loss(self):
        loss = 0.0
        for module in self.dynamic_seq.modules():
            if hasattr(module, 'regularization_loss'):
                loss += module.regularization_loss()
        return loss


class DynamicWaveletAttentionIdentity(nn.Module):
    """
    动态选择深度可分离小波卷积、SCSA注意力、恒等映射的模块。
    支持输入输出通道不一致，通过1x1卷积调整。
    可替换 YOLO 中的 C3k2 或作为一般特征增强模块。
    """
    def __init__(self, c1, c2, n=1, tau=1.0, hard=False,
                 target_ratios=None, reg_weight=0.01, entropy_weight=0.01, shortcut=True):
        super().__init__()
        self.c = c2
        self.n = n
        # 只有在输入输出通道相同且 shortcut=True 时才启用残差
        self.shortcut = shortcut and (c1 == c2)

        layers = []
        for _ in range(n):
            # 三个分支：深度可分离小波卷积、SCSA注意力、恒等映射
            # 注意：这里分支的输入输出通道都是 c1（保持不变）
            branch_wavelet = DepthwiseSeparableConvWithWTConv2d(c1, c1, k=3, s=1)  # 步长1，保持尺寸
            branch_scsa = SCSA(c1)                     # 输入输出通道相同，尺寸不变
            branch_identity = nn.Identity()
            branches = [branch_wavelet, branch_scsa, branch_identity]

            dynamic_block = DynamicMultiBranch(
                in_channels=c1,
                branches=branches,
                tau=tau,
                hard=hard,
                target_ratios=target_ratios or [1/3, 1/3, 1/3],
                reg_weight=reg_weight,
                entropy_weight=entropy_weight
            )
            layers.append(dynamic_block)

        self.dynamic_seq = nn.Sequential(*layers)
        # 如果输入输出通道不同，添加1x1卷积调整
        self.conv = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        out = self.dynamic_seq(x)
        if self.shortcut:
            out = out + x
        return self.conv(out)

    def regularization_loss(self):
        """收集所有动态块的正则化损失"""
        loss = 0.0
        for module in self.dynamic_seq.modules():
            if hasattr(module, 'regularization_loss'):
                loss += module.regularization_loss()
        return loss

class DynamicWaveletIdentity(nn.Module):
    """
    动态多分支模块：深度可分离小波卷积 vs 恒等映射。
    支持输入输出通道不同（通过1x1卷积调整），并可选残差连接。
    """
    def __init__(self, c1, c2, n=1, tau=1.0, hard=False,
                 target_ratios=None, reg_weight=0.01, shortcut=True):
        super().__init__()
        self.c = c2
        self.n = n
        self.shortcut = shortcut and (c1 == c2)

        layers = []
        for _ in range(n):
            branch_wavelet = DepthwiseSeparableConvWithWTConv2d(c1, c1, k=3, s=1)  # 保持尺寸
            branch_identity = nn.Identity()
            branches = [branch_wavelet, branch_identity]
            dynamic_block = DynamicMultiBranch(
                in_channels=c1,
                branches=branches,
                tau=tau,
                hard=hard,
                target_ratios=target_ratios or [0.5, 0.5],
                reg_weight=reg_weight
            )
            layers.append(dynamic_block)

        self.dynamic_seq = nn.Sequential(*layers)
        # 通道调整（若输入输出通道不同）
        self.conv = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        out = self.dynamic_seq(x)
        if self.shortcut:
            out = out + x
        return self.conv(out)

    def regularization_loss(self):
        loss = 0.0
        for module in self.dynamic_seq.modules():
            if hasattr(module, 'regularization_loss'):
                loss += module.regularization_loss()
        return loss



# 测试代码
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 创建模块，输入输出通道相同（64）
    block = DynamicWaveletIdentity(64, 64, n=1, tau=1.0, hard=False,
                                   target_ratios=[0.6, 0.4], reg_weight=0.01).to(device)
    x = torch.randn(2, 64, 64, 64).to(device)
    out = block(x)
    reg_loss = block.regularization_loss()
    print(f"Output shape: {out.shape}")
    print(f"Regularization loss: {reg_loss.item():.6f}")

    # 梯度测试
    loss = out.mean()
    loss.backward()
    print("Backward pass completed.")

# ========== 测试代码 ==========
# if __name__ == '__main__':
#     torch.manual_seed(42)
#     # 测试动态三分支模块
#     block = DynamicWaveletAttentionIdentity(
#         c1=64, c2=64, n=2,
#         tau=1.0, hard=False,
#         target_ratios=[0.4, 0.4, 0.2],
#         reg_weight=0.01,
#         entropy_weight=0.01,
#         shortcut=True
#     )
#     x = torch.randn(2, 64, 64, 64)
#     out = block(x)
#     reg_loss = block.regularization_loss()
#     print("输出形状:", out.shape)
#     print("正则化损失:", float(reg_loss))



# # 测试代码
# if __name__ == '__main__':
#     torch.manual_seed(42)
#     block = DynamicPartialIdentity(c1=64, c2=64, n=2, split_ratio=0.5,
#                                    tau=1.0, hard=False,
#                                    target_ratios=[0.6, 0.4], reg_weight=0.01,
#                                    entropy_weight=0.01)
#     x = torch.randn(2, 64, 64, 64)
#     out = block(x)
#     reg_loss = block.regularization_loss()
#     print("输出形状:", out.shape)
#     print("正则化损失:", float(reg_loss))