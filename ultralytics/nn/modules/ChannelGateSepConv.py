import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelGateSepConv(nn.Module):
    """
    通道分离卷积模块 —— 即插即用替换 C3k2。

    核心步骤：
        1. 通道门控：基于全局信息预测每个通道属于“目标组”的概率。
        2. 软分离：将特征按概率软分离为 target_features 与 background_features。
        3. 分组精炼：两组分别经过深度可分离卷积 + 1×1 投影。
        4. 动态融合：可学习的标量权重加权求和，加上原始输入的残差连接。

    参数:
        c1: 输入通道数
        c2: 输出通道数 (通常 c2 >= c1)
        e : 中间通道扩展比（用于门控隐藏层），默认 0.5
        target_ratio_init: 目标组通道占比的初始偏置，值越大目标组初始越大（默认 0.0 表示均匀）
    """

    def __init__(self, c1, c2, e=0.5, target_ratio_init=0.0):
        super().__init__()
        self.c1 = c1
        self.c2 = c2

        # ----- 通道门控：产生 (B, C1, 1, 1) 的通道重要性分数 -----
        hidden_ch = max(1, int(c1 * e))
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden_ch, 1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, c1, 1, bias=True)
        )
        # 初始化偏置，使初始 sigmoid 接近 0.5 + target_ratio_init 调整
        nn.init.constant_(self.gate[-1].bias, target_ratio_init)

        # ----- 分组精炼卷积（目标组）-----
        self.target_conv = nn.Sequential(
            # 深度卷积
            nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            # 1x1 投影到 c2
            nn.Conv2d(c1, c2, 1, bias=False),
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

        # ----- 通道熵正则化权重（仅训练时计算，推理时忽略）-----
        self.entropy_weight = 0.001   # 可在训练中动态调整

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 通道门控概率 (B, C, 1, 1)
        gate_prob = self.gate(x).sigmoid()        # 值域 [0,1]，每个通道属于“目标组”的概率

        # 2. 软分离：目标组 = x * gate_prob，背景组 = x * (1 - gate_prob)
        target_feat = x * gate_prob
        background_feat = x * (1.0 - gate_prob)

        # 3. 分组精炼
        target_out = self.target_conv(target_feat)          # (B, c2, H, W)
        background_out = self.background_conv(background_feat)  # (B, c2, H, W)

        # 4. 动态融合（可学习加权求和）
        w_t = self.w_target.abs() + 1e-4
        w_b = self.w_background.abs() + 1e-4
        w_sum = w_t + w_b
        out = (w_t * target_out + w_b * background_out) / w_sum

        # 5. 残差连接
        residual = self.residual(x)
        out = out + residual

        # 6. 通道熵正则项（仅在训练时返回，用于防止门控退化）
        if self.training:
            # 计算每个通道的平均门控概率
            avg_gate = gate_prob.mean(dim=(2, 3))   # (B, C)
            # 熵: H = -p*log(p) - (1-p)*log(1-p)
            eps = 1e-6
            entropy = - (avg_gate * torch.log(avg_gate + eps) +
                         (1 - avg_gate) * torch.log(1 - avg_gate + eps)).mean()
            # 我们希望熵不要太小（即门控不要过于极端），因此用负熵惩罚
            # 损失项 = - entropy * weight （加到总损失中，因为最小化总损失会最大化熵）
            # 但通常我们激励高熵，所以返回 -entropy 乘以一个权重，调用者将其加到损失上
            entropy_reg = -entropy * self.entropy_weight
        else:
            entropy_reg = None

        return out, entropy_reg, gate_prob


# ===================== 测试 =====================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 模拟输入：P2 层特征 (batch=2, channels=128, height=160, width=160)
    x = torch.randn(2, 128, 160, 160).to(device)

    # 创建模块：输入128，输出256（模拟替换 C3k2(128,256) 的场景）
    model = ChannelGateSepConv(c1=128, c2=256, e=0.5).to(device)

    # 训练模式，测试熵正则项
    model.train()
    out, entropy_reg, gate_prob = model(x)

    print(f"Input shape : {x.shape}")
    print(f"Output shape: {out.shape}")   # 期望 (2, 256, 160, 160)
    print(f"Gate prob shape: {gate_prob.shape}")  # (2, 128, 1, 1)
    print(f"Entropy reg (training): {entropy_reg.item():.5f}")

    # 推理模式，不应返回 entropy_reg（为 None）
    model.eval()
    with torch.no_grad():
        out_infer, entropy_reg_eval, _ = model(x)
    print(f"Inference output shape: {out_infer.shape}")
    print(f"Entropy reg (inference): {entropy_reg_eval}")  # 应为 None

    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 验证形状正确
    assert out.shape == (2, 256, 160, 160), f"Shape mismatch: {out.shape}"
    print("✅ All tests passed.")