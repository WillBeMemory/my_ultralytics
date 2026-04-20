import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.DeformableConv2d import DeformableConv2d
from ultralytics.nn.modules.PCCA import PCCA
from ultralytics.nn.modules.PSCA import PSCA
from ultralytics.nn.modules.SCSA import SCSA
from ultralytics.nn.modules.CoordinateAttention import CoordAtt
from ultralytics.nn.modules.DepthwiseSeparableConv import DepthwiseSeparableConvWithWTConv2d

from ultralytics.nn.modules.DepthwiseSeparableConv import DepthwiseSeparableConvWithWTConv2d
from ultralytics.nn.modules.block import Conv, DWConv
from ultralytics.nn.modules.SCSA import SCSA
from ultralytics.nn.modules.CoordinateAttention import CoordAtt
from ultralytics.nn.modules.SelfAttention import SelfAttention
from ultralytics.nn.modules.InfSA import InfSA
# from ultralytics.nn.modules.CBAM import CBAM  # 假设已实现 CBAM 模块
from ultralytics.nn.modules.RFAConv import RFAConv

__all__ = ["ConvMoE", "AttMoE","MoEBlock"]


class MoEBlockBase(nn.Module):
    """
    Mixture of Experts 基类。
    子类需要实现 _build_experts(self, channels) 方法，返回专家模块列表。
    """
    def __init__(self, c1, c2, n=1, top_k=2,shortcut=True,
                 gate_hidden=None, load_balance_weight=0.01, tau=1.0, hard=False,
                 mid_channels=None):
        super().__init__()
        self.shortcut = shortcut and (c1 == c2)
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        self.tau = tau
        self.hard = hard

        # 内部统一通道数
        if mid_channels is None:
            mid_channels = max(c1, c2)
        self.mid_channels = mid_channels

        # 输入/输出投影
        self.in_proj = nn.Conv2d(c1, mid_channels, 1) if c1 != mid_channels else nn.Identity()
        self.out_proj = nn.Conv2d(mid_channels, c2, 1) if mid_channels != c2 else nn.Identity()

        # 由子类构建专家
        self.experts = self._build_experts(mid_channels)
        num_experts = len(self.experts)

        # 门控网络
        if gate_hidden is None:
            gate_hidden = max(mid_channels // 4, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(mid_channels, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, num_experts)
        )


        self.register_buffer('expert_usage', torch.zeros(num_experts))

    def _build_experts(self, channels):
        """子类必须实现此方法，返回专家模块列表"""
        raise NotImplementedError

    def forward(self, x):
        x_proj = self.in_proj(x)
        B, C, H, W = x_proj.shape

        gate_logits = self.gate(x_proj)  # (B, E)
        if self.hard:
            # 硬化模式：直接取 top_k 索引，权重均匀分配或使用 one‑hot
            topk_indices = torch.topk(gate_logits, self.top_k, dim=-1).indices
            sparse_probs = torch.zeros_like(gate_logits)
            # 均匀权重：每个选中的专家权重 = 1 / top_k
            weight = 1.0 / self.top_k
            sparse_probs.scatter_(1, topk_indices, weight)
        else:
            # 软化模式：带温度的 softmax + top‑k 重归一化
            gate_probs = F.softmax(gate_logits / self.tau, dim=-1)
            topk_vals, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
            sparse_mask = torch.zeros_like(gate_probs).scatter_(-1, topk_indices, 1.0)
            sparse_probs = gate_probs * sparse_mask
            sparse_probs = sparse_probs / (sparse_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # 计算所有专家输出
        expert_outputs = [expert(x_proj) for expert in self.experts]

        out = torch.zeros_like(x_proj)
        for b in range(B):
            for t in range(self.top_k):
                exp_idx = topk_indices[b, t].item()
                weight = sparse_probs[b, exp_idx].item()
                out[b] += weight * expert_outputs[exp_idx][b]

        # 统计专家使用次数（仅用于负载均衡损失）
        with torch.no_grad():
            for b in range(B):
                for t in range(self.top_k):
                    self.expert_usage[topk_indices[b, t]] += 1

        out = self.out_proj(out)
        if self.shortcut:
            out = out + x
        return out

    def load_balancing_loss(self):
        # print("entering load balance")
        if self.expert_usage.sum() == 0:
            return torch.tensor(0.0, device=self.expert_usage.device)
        usage = self.expert_usage / self.expert_usage.sum()
        target = torch.ones_like(usage) / len(usage)
        loss = F.mse_loss(usage, target)
        return self.load_balance_weight * loss

    def regularization_loss(self):
        return self.load_balancing_loss()
    def reset_usage(self):
        # print("resetting usage")
        self.expert_usage.zero_()


class ConvMoE(MoEBlockBase):
    """卷积专家混合模块，专家为不同卷积变体"""
    def _build_experts(self, channels):
        return nn.ModuleList([
            DepthwiseSeparableConvWithWTConv2d(channels, channels, k=3, s=1),  # 小波卷积
            # RFAConv(channels,channels),
            # DeformableConv2d(channels, channels, kernel_size=3, modulation=True),
            # Conv(channels, channels, k=3, s=1),                               # 标准 3x3 卷积
            # DWConv(channels, channels, k=5, s=1),                             # 深度可分离 5x5
            nn.Identity()                                                     # 恒等映射
        ])


class AttMoE(MoEBlockBase):
    """注意力专家混合模块，专家均为不同类型的注意力模块"""
    def _build_experts(self, channels):
        return nn.ModuleList([

            SCSA(channels),          # 空间-通道协同注意力
            # PSCA(channels),          # 空间坐标协同
            PCCA(channels, reduction=16, stages=2),  # 通道坐标协同
            # CoordAtt(channels),      # 坐标注意力
            # SelfAttention(channels), # 自注意力
            InfSA(channels, num_steps=3, gamma=0.8),  # 新专家
            # nn.Identity()            # 恒等映射（安全网）
        ])

class MoEBlock(MoEBlockBase):
    def __init__(self, *args, experts_list=None, **kwargs):
        self.experts_list = experts_list or [nn.Identity()]
        super().__init__(*args, **kwargs)

    def _build_experts(self, channels):
        # 这里可以处理专家需要通道数的情况
        return nn.ModuleList([
            DepthwiseSeparableConvWithWTConv2d(channels, channels, k=3, s=1),  # 小波卷积
            InfSA(channels, num_steps=3, gamma=0.8),  # 新专家
            PCCA(channels, reduction=16, stages=2),  # 通道坐标协同
            # SCSA(channels),  #  空间-通道协同注意力
            nn.Identity()
        ])

class CustomMoE(MoEBlockBase):
    def __init__(self, *args, experts_list=None, **kwargs):
        self.experts_list = experts_list or [nn.Identity()]
        super().__init__(*args, **kwargs)

    def _build_experts(self, channels):
        # 这里可以处理专家需要通道数的情况
        return nn.ModuleList([
            expert(channels) if callable(expert) else expert for expert in self.experts_list
        ])


def build_moe(typ, c1, c2, **kwargs):
    """
    根据类型字符串创建对应的 MoE 子类实例。
    typ: 'att' 或 'conv' 等
    """
    if typ == 'att':
        return AttMoE(c1, c2, **kwargs)
    elif typ == 'conv':
        return ConvMoE(c1, c2, **kwargs)
    else:
        raise ValueError(f"Unknown MoE type: {typ}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 测试注意力 MoE
    att_moe = AttMoE(c1=64, c2=64, top_k=2).to(device)
    x = torch.randn(2, 64, 32, 32).to(device)
    out = att_moe(x)
    loss = att_moe.load_balancing_loss()
    print(f"AttMoE output shape: {out.shape}, load balance loss: {loss.item():.4f}")

    # 测试卷积 MoE
    conv_moe = ConvMoE(c1=64, c2=128, top_k=2, mid_channels=128).to(device)
    x2 = torch.randn(2, 64, 32, 32).to(device)
    out2 = conv_moe(x2)
    loss2 = conv_moe.load_balancing_loss()
    print(f"ConvMoE output shape: {out2.shape}, load balance loss: {loss2.item():.4f}")

    # 梯度测试
    out2.mean().backward()
    print("Backward pass completed.")