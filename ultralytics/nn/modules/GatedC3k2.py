import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv, Bottleneck, C3k, PSABlock

__all__ = ['GatedC3k2']

class GatedC3k2(nn.Module):
    """
    C3k2 with deterministic channel-wise gating.
    Instead of fixed split (half-half), each channel learns to weight the two paths:
    - Path A: through self.m (Bottleneck/C3k/PSABlock)
    - Path B: identity
    The weights are learned per channel and applied via softmax (or hard argmax).
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
        # Gating specific parameters
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

        # Input projection to c2
        self.cv1 = Conv(c1, c2, 1, 1)

        # Build the main branch (m) as a sequence of n blocks
        # Each block should output c2 channels (same as input)
        self.branch_m = nn.ModuleList()
        for _ in range(n):
            if attn:
                # PSABlock: expects input channels = c2, output = c2
                block = PSABlock(c2, attn_ratio=0.5, num_heads=max(c2 // 64, 1))
            elif c3k:
                # C3k expects input/output = c2, and internal hidden = int(c2*e)
                # We'll use n=2 (like original C3k2) but you can adjust
                block = C3k(c2, c2, n=2, shortcut=shortcut, g=g, e=e, k=3)
            else:
                # Bottleneck: input/output = c2, hidden = int(c2*e)
                block = Bottleneck(c2, c2, shortcut=shortcut, g=g, k=(3,3), e=e)
            self.branch_m.append(block)

        # Gating logits: each channel has 2 values (for m_path and identity_path)
        self.logits = nn.Parameter(torch.randn(c2, 2) * 0.01)
        self.register_buffer('target_ratios', torch.tensor([target_ratio, 1 - target_ratio]))

        # Output projection (optional, but original C3k2 has cv2)
        self.cv2 = Conv(c2, c2, 1, 1) if c2 != c1 else nn.Identity()

    def forward(self, x):
        # Project to c2
        x = self.cv1(x)  # (B, c2, H, W)

        # Branch m: apply each block sequentially
        out_m = x
        for block in self.branch_m:
            out_m = block(out_m)
        # Branch identity
        out_id = x

        # Gating: compute per-channel weights (B, c2, 2)
        B, C, H, W = x.shape
        logits = self.logits.unsqueeze(0).expand(B, -1, -1)  # (B, C, 2)
        probs = F.softmax(logits / self.tau, dim=-1)        # (B, C, 2)

        if self.hard:
            hard_probs = torch.zeros_like(probs).scatter_(2, probs.argmax(dim=-1, keepdim=True), 1.0)
            probs = (hard_probs - probs).detach() + probs

        # Weighted combination
        probs_exp = probs.unsqueeze(-1).unsqueeze(-1)  # (B, C, 2, 1, 1)
        out = (probs_exp[:, :, 0] * out_m + probs_exp[:, :, 1] * out_id)

        # Final conv
        out = self.cv2(out)
        return out

    def get_gate_probs(self):
        with torch.no_grad():
            probs = F.softmax(self.logits / self.tau, dim=-1)  # (C, 2)
            probs_mean = probs.mean(dim=0)  # (2,)
        return probs_mean

    def regularization_loss(self):
        """KL divergence to enforce target ratio of m_path usage."""
        probs = F.softmax(self.logits / self.tau, dim=-1)
        current_probs = probs.mean(dim=0)   # (2,)
        target = self.target_ratios.to(current_probs.device)
        kl = (current_probs * (torch.log(current_probs + 1e-8) - torch.log(target + 1e-8))).sum()
        return self.reg_weight * kl


# 简单的测试代码
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 参数
    c1, c2 = 64, 64
    batch, h, w = 2, 32, 32
    x = torch.randn(batch, c1, h, w).to(device)

    # 创建模块
    model = GatedC3k2(c1, c2, n=2, c3k=False, e=0.5, attn=False, g=1, shortcut=True,
                      tau=5.0, hard=False, reg_weight=0.01, target_ratio=0.5).to(device)
    print(model)

    # 前向传播
    out = model(x)
    print(f"Output shape: {out.shape}")

    # 正则化损失
    reg_loss = model.regularization_loss()
    print(f"Regularization loss: {reg_loss.item():.6f}")

    # 总损失反向传播
    loss = out.mean() + reg_loss
    loss.backward()
    print("Backward pass completed.")
    print("All tests passed!")