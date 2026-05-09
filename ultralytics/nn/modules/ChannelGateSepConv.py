import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ECA(nn.Module):
    """高效通道注意力（Efficient Channel Attention）"""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # 自适应卷积核大小
        t = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, 1, 1) 从池化得到
        y = x.squeeze(-1).transpose(-1, -2)      # (B, 1, C)
        y = self.conv(y)                         # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)    # (B, C, 1, 1)
        return self.sigmoid(y)


class ChannelGateSepConv(nn.Module):
    """
    通道分离卷积模块 —— 即插即用替换 C3k2。

    参数 (与 C3k2 完全对齐):
        c1                : 输入通道数
        c2                : 输出通道数
        n                 : 占位参数 (兼容 C3k2 接口，未实际使用)
        c3k               : 是否在目标支路启用 ECA 通道注意力强化
                            True  → 目标支路精炼后接 ECA，背景支路保持轻量
                            False → 目标/背景支路均仅使用轻量深度可分离卷积
        e                 : 扩展比，用于门控隐藏层
        target_ratio_init : 初始目标组通道占比 (0~1)，小目标推荐 0.12
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, target_ratio_init=0.12):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.c3k = c3k
        self.e = e

        if 0 < target_ratio_init < 1:
            gate_bias = math.log(target_ratio_init / (1 - target_ratio_init))
        else:
            raise ValueError("target_ratio_init should be in (0, 1)")

        # ----- 通道门控 (B, C1, 1, 1) -----
        hidden_ch = max(1, int(c1 * e))
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden_ch, 1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, c1, 1, bias=True)
        )
        nn.init.constant_(self.gate[-1].bias, gate_bias)

        # ----- 基础分组精炼块（目标/背景支路共用）-----
        self.target_conv = self._make_group_conv(c1, c2)
        self.background_conv = self._make_group_conv(c1, c2)

        # ----- 强化分支：用 ECA 替代笨重的 Bottleneck -----
        if self.c3k:
            # 目标支路：精炼后通过 ECA 自适应增强重要通道
            self.target_bn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ECA(c2)
            )
        else:
            self.target_bn = nn.Identity()
        # 背景支路永远保持轻量
        self.background_bn = nn.Identity()

        # ----- 动态融合权重 -----
        self.w_target = nn.Parameter(torch.tensor(0.5))
        self.w_background = nn.Parameter(torch.tensor(0.5))

        # ----- 残差投影 -----
        self.residual = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

        # ----- 熵正则 -----
        self.entropy_weight = 0.001
        self.entropy_reg = None

    def _make_group_conv(self, cin, cout):
        """分组精炼基础块：深度卷积 + 1x1 投影"""
        return nn.Sequential(
            nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False),
            nn.BatchNorm2d(cin),
            nn.SiLU(inplace=True),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 通道门控
        gate_prob = self.gate(x).sigmoid()

        # 2. 软分离
        target_feat = x * gate_prob
        background_feat = x * (1.0 - gate_prob)

        # 3. 分组基础精炼
        target_out = self.target_conv(target_feat)
        background_out = self.background_conv(background_feat)

        # 4. 目标支路可选的 ECA 增强（背景支路直接通过）
        if self.c3k:
            # ECA 返回通道权重 (B, C2, 1, 1)，乘到目标特征上
            attn_weight = self.target_bn(target_out)  # (B, C2, 1, 1)
            target_out = target_out * attn_weight
        # 背景支路不需要额外操作

        # 5. 动态融合
        w_t = self.w_target.abs() + 1e-4
        w_b = self.w_background.abs() + 1e-4
        w_sum = w_t + w_b
        out = (w_t * target_out + w_b * background_out) / w_sum

        # 6. 残差连接
        out = out + self.residual(x)

        # 7. 熵正则（仅训练）
        if self.training:
            avg_gate = gate_prob.mean(dim=(2, 3))
            eps = 1e-6
            entropy = - (avg_gate * torch.log(avg_gate + eps) +
                         (1 - avg_gate) * torch.log(1 - avg_gate + eps)).mean()
            self.entropy_reg = -entropy * self.entropy_weight
        else:
            self.entropy_reg = None

        return out


# ===================== 测试 =====================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 模拟输入
    x = torch.randn(2, 128, 160, 160).to(device)

    # c3k=False（轻量，无 ECA）
    light = ChannelGateSepConv(128, 256, n=1, c3k=False, e=0.5).to(device)
    y_light = light(x)
    print(f"Light (c3k=False): in {x.shape} -> out {y_light.shape}")

    # c3k=True（目标支路加 ECA）
    strong = ChannelGateSepConv(128, 256, n=1, c3k=True, e=0.5).to(device)
    y_strong = strong(x)
    print(f"Strong (c3k=True): in {x.shape} -> out {y_strong.shape}")

    # 参数量对比
    params_light = sum(p.numel() for p in light.parameters())
    params_strong = sum(p.numel() for p in strong.parameters())
    print(f"Params: light {params_light:,} | strong {params_strong:,}")

    # 背景支路类型（恒为 Identity）
    print(f"Light background_bn: {type(light.background_bn)}")
    print(f"Strong background_bn: {type(strong.background_bn)}")

    # 目标支路强模式下应为 Sequential（包含 ECA）
    print(f"Strong target_bn: {type(strong.target_bn)}")