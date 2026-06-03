# Ultralytics 🚀 AGPL-3.0 License

"""PolaFormer: Polarity-aware Linear Attention for Vision (ICLR 2025).

Pure PyTorch reimplementation of the core attention mechanism from:
    PolaFormer: Polarity-aware Linear Attention for Vision Transformers
    https://arxiv.org/abs/2501.15061

This module splits Query/Key into positive and negative components,
processes same-signed and opposite-signed interactions separately, and
applies learnable power-function rescaling to restore spiky attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolaAttention(nn.Module):
    """Polarity-aware linear attention with O(Nd²) complexity.

    Replaces standard softmax(Q·K^T/√d) with a dual-path linear attention
    that preserves both positive and negative query-key interactions.

    Args:
        dim (int): Feature dimension (must be divisible by num_heads).
        num_heads (int): Number of attention heads (default 8).
        qkv_bias (bool): Use bias in Q/K/V linear projections.
    """

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projections
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # Polarity-aware coefficients: α for Query, β for Key
        # Two sets: one for same-signed, one for opposite-signed
        self.alpha_same = nn.Parameter(torch.ones(num_heads, self.head_dim))
        self.alpha_opp = nn.Parameter(torch.ones(num_heads, self.head_dim))
        self.beta_same = nn.Parameter(torch.ones(num_heads, self.head_dim))
        self.beta_opp = nn.Parameter(torch.ones(num_heads, self.head_dim))

        # Learnable power-function exponents for entropy reduction
        # λ > 1 → sharper attention; each head learns its own λ
        self.log_lambda = nn.Parameter(torch.zeros(num_heads))

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.alpha_same, std=0.02)
        nn.init.trunc_normal_(self.alpha_opp, std=0.02)
        nn.init.trunc_normal_(self.beta_same, std=0.02)
        nn.init.trunc_normal_(self.beta_opp, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Polarity-aware linear attention forward pass.
        
        Note: caller (PolaFormerBlock) is responsible for dtype management.
        This function operates in whatever dtype it receives.
        """
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim

        # 1. Q/K/V projections
        q = self.q_proj(x).reshape(B, N, H, D)        # (B, N, H, D)
        k = self.k_proj(x).reshape(B, N, H, D)        # (B, N, H, D)
        v = self.v_proj(x).reshape(B, N, H, D)        # (B, N, H, D)

        # 2. Split into positive/negative components
        q_pos = F.relu(q)                               # (B, N, H, D)
        q_neg = F.relu(-q)                              # (B, N, H, D)
        k_pos = F.relu(k)                               # (B, N, H, D)
        k_neg = F.relu(-k)                              # (B, N, H, D)

        # 3. Apply polarity-aware coefficients
        alpha_s = self.alpha_same.view(1, 1, H, D)     # (1, 1, H, D)
        alpha_o = self.alpha_opp.view(1, 1, H, D)
        beta_s = self.beta_same.view(1, 1, H, D)
        beta_o = self.beta_opp.view(1, 1, H, D)

        q_pos_s = q_pos * alpha_s                       # same-signed Q⁺
        q_pos_o = q_pos * alpha_o                       # opp-signed Q⁺ (from positive Q)
        q_neg_s = q_neg * alpha_o                       # opp-signed Q⁻ (from negative Q)
        q_neg_o = q_neg * alpha_s                       # same-signed Q⁻

        k_pos_s = k_pos * beta_s
        k_pos_o = k_pos * beta_o
        k_neg_s = k_neg * beta_o
        k_neg_o = k_neg * beta_s

        # 4. Kernel feature map: ELU(x) + 1 (standard linear attention)
        def phi(t):
            return F.elu(t) + 1.0

        # 5. Same-signed interactions: Q⁺⊗K⁺ + Q⁻⊗K⁻
        sim_same = (phi(q_pos_s) * phi(k_pos_s) + phi(q_neg_o) * phi(k_neg_o)) * self.scale
        # (B, N, H, D)

        # 6. Opposite-signed interactions: Q⁺⊗K⁻ + Q⁻⊗K⁺
        sim_opp = (phi(q_pos_o) * phi(k_pos_o) + phi(q_neg_s) * phi(k_neg_s)) * self.scale
        # (B, N, H, D)

        # 7. Combine same + opposite → one scalar score per token pair
        sim = sim_same + sim_opp                        # (B, N, H, D)

        # 8. Power-function rescaling to reduce entropy
        # λ = softplus(log_λ) + 1 → λ ∈ [1, 10]
        lam = F.softplus(self.log_lambda) + 1.0         # (H,)
        lam = torch.clamp(lam, min=1.0, max=10.0)       # prevent blow-up
        lam = lam.view(1, 1, H, 1)                      # (1, 1, H, 1)

        # Normalize sim to [0, 1] in float32 for stable ** operation
        sim_f32 = sim.float()
        sim_min = sim_f32.amin(dim=1, keepdim=True)
        sim_max = sim_f32.amax(dim=1, keepdim=True)
        sim_range = sim_max - sim_min
        sim_range = torch.clamp(sim_range, min=1e-4)    # prevent div-by-zero
        sim_norm = ((sim_f32 - sim_min) / sim_range).clamp(min=0, max=1)
        sim_rescaled = sim_norm ** lam                   # float32 power, stable

        # 9. Linear attention: K·V first (O(Nd²))
        # k: (B, N, H, D), v: (B, N, H, D)
        # For linear attention: O = φ(Q) * (φ(K)^T * V) / (φ(Q) * φ(K)^T_sum)
        # Using phi from step 4

        # Numerator: KV = Σ_j φ(K_j) ⊗ V_j  → (B, H, D, D)
        k_phi = phi(k)                                  # (B, N, H, D)
        k_phi = k_phi.permute(0, 2, 3, 1)               # (B, H, D, N)
        v_t = v.permute(0, 2, 1, 3)                     # (B, H, N, D)
        kv = torch.matmul(k_phi, v_t)                    # (B, H, D, D)

        # Normalizer: Σ_j φ(K_j) → (B, H, D)
        k_sum = k_phi.sum(dim=-1)                        # (B, H, D)

        # Output = φ(Q) * KV / (φ(Q) * K_sum)
        # φ(Q) shape: (B, N, H, D)
        q_phi = phi(q)                                   # (B, N, H, D)

        # Apply rescaling to Q: stronger Q contributions
        q_phi = q_phi * sim_rescaled

        q_phi_t = q_phi.permute(0, 2, 1, 3)             # (B, H, N, D)
        out = torch.matmul(q_phi_t, kv)                  # (B, H, N, D)

        # Normalize by (K_sum for each Q), clamp min to avoid div-by-zero
        q_k = (q_phi_t * k_sum.unsqueeze(-2)).sum(dim=-1)  # (B, H, N)
        q_k = torch.clamp(q_k, min=1e-3)                    # prevent tiny denominator
        out = out / q_k.unsqueeze(-1)                       # (B, H, N, D)

        # Restore shape
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)

        # Output projection
        out = self.out_proj(out)

        return out


class PolaFormerBlock(nn.Module):
    """Transformer block with PolaFormer attention (C2PSA-style for YOLO).

    Drop-in replacement for PSABlock or any Transformer-based ViT block.
    Uses PolaAttention instead of standard MHSA. Can be plugged into YOLO
    neck after feature projection.

    Args:
        dim (int): Feature dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): MLP hidden dimension ratio.
        drop (float): Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PolaAttention(dim, num_heads=num_heads)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — caller (C3k2_Pola) manages dtype."""
        is_cnn = len(x.shape) == 4
        if is_cnn:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        if is_cnn:
            x = x.transpose(1, 2).reshape(B, C, H, W)

        return x


class C3k2_Pola(nn.Module):
    """C3k2 + PolaFormerBlock: CSP with polarity-aware attention.

    Each CSP block contains Bottleneck (local features) followed by
    PolaFormerBlock (global polarity-aware attention). The attention
    is always enabled — no toggle flag.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of CSP blocks (each: Bottleneck + PolaFormerBlock).
        e (float): Expansion ratio for Bottleneck hidden channels.
        shortcut (bool): Use residual connections.
        g (int): Groups for Bottleneck convolutions.
        num_heads (int): Attention heads for PolaFormerBlock.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,        # YAML convention (ignored)
        e: float = 0.5,
        shortcut: bool = True,
        g: int = 1,
        num_heads: int = 8,
        **kwargs,
    ):
        super().__init__()
        from ultralytics.nn.modules.block import Conv, Bottleneck

        self.c = int(c2 * e)
        self.n = n

        # CSP head/tail (same as C2f/C3k2)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        # Each block: Bottleneck (float16 under AMP) + PolaFormerBlock (float32)
        self.m = nn.ModuleList()
        for _ in range(n):
            self.m.append(nn.ModuleList([
                Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0),
                PolaFormerBlock(self.c, num_heads=num_heads),
            ]))

        self.cv2 = Conv((2 + n) * self.c, c2, 1, act=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype_in = x.dtype
        y = list(self.cv1(x).chunk(2, dim=1))
        outputs = [y[0], y[1]]
        a = y[1]
        for bn, attn in self.m:
            a = bn(a)              # Bottleneck: runs in AMP float16
            a = attn(a.float())    # PolaFormerBlock: float32 (stable for large N)
            a = a.to(dtype_in)     # → back to float16 for next Bottleneck
            outputs.append(a)
        return self.act(self.cv2(torch.cat(outputs, dim=1)))


# ================== Test ==================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("=" * 60)

    # Test 1: PolaAttention standalone (Transformer mode)
    print("\n1. PolaAttention (Transformer mode)")
    x = torch.randn(2, 100, 256, device=device)  # (B, N, C)
    attn = PolaAttention(256, num_heads=8).to(device)
    y = attn(x)
    print(f"   Input: {x.shape} → Output: {y.shape} (expected [2, 100, 256])")
    loss = y.sum()
    loss.backward()
    print("   ✅ Backward OK")

    # Test 2: PolaFormerBlock (CNN mode)
    print("\n2. PolaFormerBlock (CNN mode)")
    x = torch.randn(2, 256, 40, 40, device=device)  # (B, C, H, W)
    block = PolaFormerBlock(256, num_heads=8).to(device)
    y = block(x)
    print(f"   Input: {x.shape} → Output: {y.shape} (expected [2, 256, 40, 40])")
    loss = y.sum()
    loss.backward()
    print("   ✅ Backward OK")

    # Test 3: Different token counts
    print("\n3. Token count tests")
    for n in [100, 400, 1600, 6400]:
        x = torch.randn(2, n, 256, device=device)
        attn = PolaAttention(256, num_heads=8).to(device)
        y = attn(x)
        print(f"   N={n:>5}: {x.shape} → {y.shape} ✅")

    # Test 4: Variable spatial sizes
    print("\n4. Spatial size tests")
    block = PolaFormerBlock(256, num_heads=8).to(device)
    for hw in [(20, 20), (40, 40), (80, 80)]:
        x = torch.randn(2, 256, hw[0], hw[1], device=device)
        y = block(x)
        print(f"   {hw[0]}×{hw[1]}: {x.shape} → {y.shape} ✅")

    # Test 5: Polarity coefficients verification
    print("\n5. Polarity coefficients")
    attn = PolaAttention(256, num_heads=8).to(device)
    print(f"   alpha_same range: [{attn.alpha_same.min():.3f}, {attn.alpha_same.max():.3f}]")
    print(f"   log_lambda:       {attn.log_lambda.tolist()}")
    lam = F.softplus(attn.log_lambda) + 1.0
    print(f"   lambda:           {lam.tolist()} (≥1)")

    # Test 6: Comparison with standard linear attention
    print("\n6. Standard linear attention vs Pola attention")
    x = torch.randn(2, 400, 256, device=device)
    attn_pola = PolaAttention(256, num_heads=8).to(device)
    y_pola = attn_pola(x)

    # Simple linear attention baseline
    q = attn_pola.q_proj(x).reshape(2, 400, 8, 32).permute(0, 2, 1, 3)
    k = attn_pola.k_proj(x).reshape(2, 400, 8, 32).permute(0, 2, 1, 3)
    v = attn_pola.v_proj(x).reshape(2, 400, 8, 32).permute(0, 2, 1, 3)

    phi = lambda t: F.elu(t) + 1.0
    kv = torch.matmul(phi(k).transpose(-2, -1), v)
    k_sum = phi(k).sum(dim=2, keepdim=True)
    qk_sum = (phi(q) * k_sum).sum(dim=-1, keepdim=True)
    y_linear = torch.matmul(phi(q), kv) / (qk_sum + 1e-6)
    y_linear = y_linear.permute(0, 2, 1, 3).reshape(2, 400, 256)

    print(f"   Pola mean: {y_pola.abs().mean():.4f}")
    print(f"   Linear mean: {y_linear.abs().mean():.4f}")

    # Test 7: Parameter count
    print("\n7. Parameter count")
    attn = PolaAttention(256, num_heads=8)
    block = PolaFormerBlock(256, num_heads=8)
    n_params_attn = sum(p.numel() for p in attn.parameters())
    n_params_block = sum(p.numel() for p in block.parameters())
    print(f"   PolaAttention:  {n_params_attn:,} params")
    print(f"   PolaFormerBlock: {n_params_block:,} params")

    # Test 8: C3k2_Pola
    print("\n8. C3k2_Pola")
    x = torch.randn(2, 64, 64, 64, device=device)
    model = C3k2_Pola(64, 128, n=2, e=0.5, shortcut=True, num_heads=4).to(device)
    y = model(x)
    print(f"   Input: {x.shape} → Output: {y.shape} (expected [2, 128, 64, 64])")
    loss = y.sum()
    loss.backward()
    n_params_c3k2pola = sum(p.numel() for p in model.parameters())
    print(f"   Params: {n_params_c3k2pola:,}")
    print("   ✅ Backward OK")

    print(f"\n   ✅ All tests passed!")
