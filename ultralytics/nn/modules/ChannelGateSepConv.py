import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelGateSepConv(nn.Module):
    """
    通道分离卷积模块 —— 即插即用替换 C3k2。
    返回值: 前向传播输出 (Tensor)。
    训练时额外保存 entropy_reg 于 self.entropy_reg 属性，可在外部提取用于损失计算。
    """

    def __init__(self, c1, c2, n=1, e=0.5, target_ratio_init=0.12):
        super().__init__()
        self.c1 = c1
        self.c2 = c2

        # 将目标初始概率转换为门控输出层的偏置（logit）
        if 0 < target_ratio_init < 1:
            gate_bias = math.log(target_ratio_init / (1 - target_ratio_init))
        else:
            raise ValueError("target_ratio_init should be in (0, 1), e.g., 0.12")

        # ----- 通道门控：产生 (B, C1, 1, 1) 的通道重要性分数 -----
        hidden_ch = max(1, int(c1 * e))
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden_ch, 1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, c1, 1, bias=True)
        )
        nn.init.constant_(self.gate[-1].bias, gate_bias)

        # ----- 分组精炼卷积（目标组）-----
        self.target_conv = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False),  # 深度卷积
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c2, 1, bias=False),                         # 1x1 投影
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )

        # ----- 分组精炼卷积（背景组）-----
        self.background_conv = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )

        # ----- 动态融合权重：两个可学习标量 -----
        self.w_target = nn.Parameter(torch.tensor(0.5))
        self.w_background = nn.Parameter(torch.tensor(0.5))

        # ----- 残差投影（当 c1 != c2 时调整通道）-----
        self.residual = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

        # ----- 熵正则参数 -----
        self.entropy_weight = 0.001
        self.entropy_reg = None   # 训练时存储正则项

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 通道门控概率 (B, C, 1, 1)
        gate_prob = self.gate(x).sigmoid()

        # 2. 软分离
        target_feat = x * gate_prob
        background_feat = x * (1.0 - gate_prob)

        # 3. 分组精炼
        target_out = self.target_conv(target_feat)
        background_out = self.background_conv(background_feat)

        # 4. 动态融合
        w_t = self.w_target.abs() + 1e-4
        w_b = self.w_background.abs() + 1e-4
        w_sum = w_t + w_b
        out = (w_t * target_out + w_b * background_out) / w_sum

        # 5. 残差连接
        out = out + self.residual(x)

        # 6. 计算熵正则并保存（仅训练时）
        if self.training:
            avg_gate = gate_prob.mean(dim=(2, 3))   # (B, C)
            eps = 1e-6
            entropy = - (avg_gate * torch.log(avg_gate + eps) +
                         (1 - avg_gate) * torch.log(1 - avg_gate + eps)).mean()
            self.entropy_reg = -entropy * self.entropy_weight
        else:
            self.entropy_reg = None

        return out   # 必须返回单个 Tensor