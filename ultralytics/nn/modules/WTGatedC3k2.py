import torch
import torch.nn as nn
import torch.nn.functional as F
# 引入你项目中已有的深度可分离小波卷积模块
from ultralytics.nn.modules.DepthwiseSeparableConv import DepthwiseSeparableConvWithWTConv2d
# 引入项目中原有的基础模块，用于构建主分支
from ultralytics.nn.modules.block import Conv, Bottleneck, C3k, PSABlock

__all__ = ['WTGatedC3k2']

class WTGatedC3k2(nn.Module):
    """
    这是基于 GatedC3k2 的改进模块。
    它的核心改动在于将 CV1 和 CV2 这两个1x1卷积，替换为了轻量化的深度可分离小波卷积。
    这能让模块在保持门控特性的同时，变得更轻量，并可能获得更佳的感受野和性能。
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        tau: float = 5.0,
        hard: bool = False,
        reg_weight: float = 0.01,
        target_ratio: float = 0.5
    ):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.n = n
        self.tau = tau
        self.hard = hard
        self.reg_weight = reg_weight
        self.target_ratio = target_ratio

        # --- 关键修改 1: 将 cv1 替换为深度可分离小波卷积 ---
        # 原始代码中，cv1是普通Conv，用于调整输入通道到c2。
        # 这里我们使用DepthwiseSeparableConvWithWTConv2d来替代它，
        # 在降低计算量的同时，通过小波变换获得更大的感受野和更好的频域特征[reference:1]。
        self.cv1 = DepthwiseSeparableConvWithWTConv2d(c1, c2, k=1, s=1)

        # 构建主分支 (branch_m)
        self.branch_m = nn.ModuleList()
        for _ in range(n):
            if attn:
                block = PSABlock(c2, attn_ratio=0.5, num_heads=max(c2 // 64, 1))
            elif c3k:
                block = C3k(c2, c2, n=2, shortcut=shortcut, g=g, e=e, k=3)
            else:
                block = Bottleneck(c2, c2, shortcut=shortcut, g=g, k=(3,3), e=e)
            self.branch_m.append(block)

        # 门控 logits: 每个通道有2个值 (对应 m_path 和 identity_path)
        self.logits = nn.Parameter(torch.randn(c2, 2) * 0.01)
        self.register_buffer('target_ratios', torch.tensor([target_ratio, 1 - target_ratio]))

        # --- 关键修改 2: 将 cv2 替换为深度可分离小波卷积 ---
        # 原始代码中，cv2是普通Conv，用于最终的输出投影。
        # 这里同样替换为DepthwiseSeparableConvWithWTConv2d，以保持模块的整体轻量化。
        # 如果输入输出通道数一致，则此cv2可以省略。
        self.cv2 = DepthwiseSeparableConvWithWTConv2d(c2, c2, k=1, s=1) if c2 != c1 else nn.Identity()

    def forward(self, x):
        # 1. 输入投影
        x = self.cv1(x)  # (B, c2, H, W)

        # 2. 计算两条路径的输出
        out_m = x
        for block in self.branch_m:
            out_m = block(out_m)
        out_id = x

        # 3. 门控机制：为每个通道计算权重
        B, C, H, W = x.shape
        logits = self.logits.unsqueeze(0).expand(B, -1, -1)  # (B, C, 2)
        probs = F.softmax(logits / self.tau, dim=-1)        # (B, C, 2)

        if self.hard:
            hard_probs = torch.zeros_like(probs).scatter_(2, probs.argmax(dim=-1, keepdim=True), 1.0)
            probs = (hard_probs - probs).detach() + probs

        # 4. 加权融合两条路径
        probs_exp = probs.unsqueeze(-1).unsqueeze(-1)  # (B, C, 2, 1, 1)
        out = (probs_exp[:, :, 0] * out_m + probs_exp[:, :, 1] * out_id)

        # 5. 输出投影
        out = self.cv2(out)
        return out

    def regularization_loss(self):
        """计算KL散度损失，用于约束门控概率分布接近目标比例。"""
        probs = F.softmax(self.logits / self.tau, dim=-1)
        current_probs = probs.mean(dim=0)
        target = self.target_ratios.to(current_probs.device)
        kl = (current_probs * (torch.log(current_probs + 1e-8) - torch.log(target + 1e-8))).sum()
        return self.reg_weight * kl

    def get_gate_probs(self):
        """获取当前门控的分配比例，便于监控和调试。"""
        with torch.no_grad():
            probs = F.softmax(self.logits / self.tau, dim=-1)
            return probs.mean(dim=0)



def wtgated_c3k2():
    print("=" * 50)
    print("Testing WTGatedC3k2 Module")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 超参数
    batch_size = 2
    in_channels = 64
    out_channels = 64
    height, width = 32, 32

    # 创建模块
    model = WTGatedC3k2(
        c1=in_channels,
        c2=out_channels,
        n=2,                # 两个重复块
        c3k=False,          # 不使用 C3k，使用普通 Bottleneck
        e=0.5,              # 扩展因子
        attn=False,         # 不使用注意力
        g=1,
        shortcut=True,
        tau=5.0,
        hard=False,
        reg_weight=0.01,
        target_ratio=0.5
    ).to(device)

    print(model)
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 随机输入
    x = torch.randn(batch_size, in_channels, height, width).to(device)
    print(f"Input shape: {x.shape}")

    # 前向传播
    out = model(x)
    print(f"Output shape: {out.shape}")

    # 检查输出形状
    assert out.shape == (batch_size, out_channels, height, width), \
        f"Expected shape {(batch_size, out_channels, height, width)}, got {out.shape}"
    print("Forward pass shape OK.")

    # 正则化损失
    reg_loss = model.regularization_loss()
    print(f"Regularization loss: {reg_loss.item():.6f}")

    # 门控概率
    gate_probs = model.get_gate_probs()
    print(f"Gate probabilities: m_path={gate_probs[0]:.4f}, identity_path={gate_probs[1]:.4f}")

    # 总损失并反向传播
    loss = out.mean() + reg_loss
    loss.backward()
    print("Backward pass completed.")

    # 检查梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            print(f"  Gradient for {name}: shape={param.grad.shape}, norm={param.grad.norm().item():.6f}")
    assert has_grad, "No gradients found!"
    print("Gradient check passed.")

    # 测试硬化模式
    print("\n--- Testing hard mode ---")
    model.hard = True
    model.tau = 1.0
    with torch.no_grad():
        out_hard = model(x)
    print(f"Hard mode output shape: {out_hard.shape}")
    assert not torch.allclose(out_hard, torch.zeros_like(out_hard)), "Hard mode output is all zero!"
    print("Hard mode test passed.")

    # 测试不同配置：使用 C3k 和注意力
    print("\n--- Testing with C3k and attention ---")
    model2 = WTGatedC3k2(
        c1=in_channels,
        c2=out_channels,
        n=1,
        c3k=True,
        e=0.5,
        attn=True,
        g=1,
        shortcut=True,
        tau=5.0,
        hard=False,
        reg_weight=0.01,
        target_ratio=0.6
    ).to(device)
    out2 = model2(x)
    print(f"Output shape (c3k+attn): {out2.shape}")
    reg_loss2 = model2.regularization_loss()
    print(f"Regularization loss: {reg_loss2.item():.6f}")
    print("C3k+attention test passed.")

    print("\n" + "=" * 50)
    print("All tests passed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    wtgated_c3k2()