import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.DepthwiseSeparableConv import DepthwiseSeparableConvWithWTConv2d
from ultralytics.nn.modules.MoE import ConvMoE, AttMoE

__all__ = ['TernaryMoEBlock']

class PruneBranch(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)

class TernaryMoEBlock(nn.Module):
    def __init__(self,
                 c1,                     # 输入通道
                 c2,                     # 输出通道
                 n=1,                    # 重复次数（兼容 YOLO，此处不使用）
                 target_ratios=None,    # 三分支目标比例 [prune, moe, conv]
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 groups=1,
                 tau=1.0,
                 hard=False,
                 reg_weight=0.01,
                 moe_type='att',
                 moe_kwargs=None,
                 shortcut=True):
        super().__init__()
        assert c1 == c2, "TernaryMoEBlock requires c1 == c2"
        self.c1 = c1
        self.c2 = c2
        self.n = n          # 仅用于兼容，不实际使用
        self.stride = stride
        self.tau = tau
        self.hard = hard
        self.reg_weight = reg_weight
        self.shortcut = shortcut and (stride == 1)

        if target_ratios is None:
            target_ratios = [0.2, 0.4, 0.4]  # prune, moe, conv
        self.register_buffer('target_ratios', torch.tensor(target_ratios))

        # 初始化 logits（每个通道独立）
        logits_init = torch.log(torch.tensor(target_ratios) + 1e-8)
        logits_init = self.tau * logits_init
        self.logits = nn.Parameter(logits_init.unsqueeze(0).repeat(self.c1, 1))

        if moe_kwargs is None:
            moe_kwargs = {}
        # 三个分支
        self.prune_branch = PruneBranch()
        self.moe_branch = AttMoE(c1, c1, **moe_kwargs)
        self.conv_branch = DepthwiseSeparableConvWithWTConv2d(
            c1, c2,
            k=kernel_size, s=stride, p=padding, g=groups
        )

        self.register_buffer('current_probs', torch.zeros(3))

    def forward(self, x):
        B, C, H, W = x.shape
        logits = self.logits.unsqueeze(0).expand(B, -1, -1)   # (B, C, 3)
        gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8))
        y = logits + gumbel
        probs = F.softmax(y / self.tau, dim=-1)

        if self.hard:
            hard_probs = torch.zeros_like(probs).scatter_(2, probs.argmax(dim=-1, keepdim=True), 1.0)
            probs = (hard_probs - probs).detach() + probs

        out_prune = self.prune_branch(x)
        out_moe = self.moe_branch(x)
        out_conv = self.conv_branch(x)

        if self.stride > 1:
            out_prune = F.avg_pool2d(out_prune, kernel_size=self.stride, stride=self.stride)
            out_moe = F.avg_pool2d(out_moe, kernel_size=self.stride, stride=self.stride)

        probs_exp = probs.unsqueeze(-1).unsqueeze(-1)
        out = (probs_exp[:, :, 0] * out_prune +
               probs_exp[:, :, 1] * out_moe +
               probs_exp[:, :, 2] * out_conv)

        if self.shortcut:
            out = out + x

        with torch.no_grad():
            self.current_probs = probs.mean(dim=(0, 1))

        return out

    def regularization_loss(self):
        probs = F.softmax(self.logits / self.tau, dim=-1)
        current_probs = probs.mean(dim=0)
        target = self.target_ratios.to(current_probs.device)
        kl = (current_probs * (torch.log(current_probs + 1e-8) - torch.log(target + 1e-8))).sum()
        ternary_loss = self.reg_weight * kl
        moe_loss = self.moe_branch.regularization_loss() if hasattr(self.moe_branch, 'regularization_loss') else 0.0
        return ternary_loss + moe_loss

    def reset_usage(self):
        if hasattr(self.moe_branch, 'reset_usage'):
            self.moe_branch.reset_usage()

def ternary_moe_block():
    print("="*50)
    print("Testing TernaryMoEBlock")
    print("="*50)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    in_channels = 64
    batch_size = 2
    height, width = 32, 32
    target_ratios = [0.2, 0.4, 0.4]
    tau = 5.0
    hard = False
    reg_weight = 0.01
    moe_type = 'conv'
    moe_kwargs = {'top_k': 2, 'load_balance_weight': 0.01}

    model = TernaryMoEBlock(
        c1=in_channels,
        c2=in_channels,
        target_ratios=target_ratios,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        tau=tau,
        hard=hard,
        reg_weight=reg_weight,
        moe_type=moe_type,
        moe_kwargs=moe_kwargs,
        shortcut=True
    ).to(device)

    x = torch.randn(batch_size, in_channels, height, width).to(device)
    out = model(x)
    print(f"Output shape: {out.shape}")
    loss = out.mean() + model.regularization_loss()
    loss.backward()
    print("Backward OK")
    print(f"Branch probs: {model.current_probs.cpu().numpy()}")
    print("All tests passed!")

if __name__ == "__main__":
    ternary_moe_block()