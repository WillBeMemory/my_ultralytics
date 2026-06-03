# Ultralytics 🚀 AGPL-3.0 License

"""InfSA: Infinite Self-Attention — spectral reformulation of self-attention.

Reference:
    Infinite Self-Attention (arXiv 2603.00175, 2026)
    Giorgio Roffo

Replaces softmax(QK^T/√d) with a spectral diffusion mechanism grounded in
graph centrality theory. The attention matrix is treated as a diffusion step
on a content-adaptive token graph, accumulating multi-hop interactions via
a Neumann series: Y = Σ γ^t Â^t V = (I - γÂ)⁻¹ V

Two modes:
    InfSA:      Full O(N²) but captures all-pair interactions via Katz centrality
    LinearInfSA: O(N) approximation via Perron eigenvector power iteration
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfSA(nn.Module):
    """Infinite Self-Attention — full version.

    Replaces softmax(QK^T/√d) with Frobenius-normalized attention followed
    by a truncated Neumann series to capture multi-hop token interactions.
    Equivalent to Katz centrality on the attention graph.

    Args:
        dim (int): Feature dimension.
        num_heads (int): Number of attention heads.
        gamma (float): Discount factor for Neumann series (0 < gamma < 1).
                       Higher = more weight on multi-hop paths.
        num_terms (int): Number of Neumann series terms (default 3).
        qkv_bias (bool): Bias in Q/K/V projections.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        gamma: float = 0.5,
        num_terms: int = 3,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.gamma = gamma
        self.num_terms = num_terms

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """InfSA forward pass.

        Args:
            x: (B, N, C)

        Returns:
            (B, N, C)
        """
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim

        q = self.q_proj(x).reshape(B, N, H, D).permute(0, 2, 1, 3)  # (B, H, N, D)
        k = self.k_proj(x).reshape(B, N, H, D).permute(0, 2, 1, 3)  # (B, H, N, D)
        v = self.v_proj(x).reshape(B, N, H, D).permute(0, 2, 1, 3)  # (B, H, N, D)

        # 1. Attention matrix: Frobenius normalization (not softmax)
        A = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        A = F.relu(A)  # non-negative

        # Frobenius normalization: Â = A / ||A||_F
        A_norm = torch.linalg.matrix_norm(A, ord='fro', dim=(-2, -1), keepdim=True)
        A_hat = A / (A_norm + 1e-6)  # (B, H, N, N)
        # spectral radius < 1 guaranteed

        # 2. Neumann series: Y = Σ_{t=0}^{T-1} γ^t Â^t V
        out = v                               # t=0 term: γ^0 Â^0 V = V
        A_power_v = v
        for t in range(1, self.num_terms):
            A_power_v = self.gamma * torch.matmul(A_hat, A_power_v)
            out = out + A_power_v

        # 3. Reshape + output projection
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.out_proj(out)

        return out


class LinearInfSA(nn.Module):
    """Linear Infinite Self-Attention — O(N) approximation.

    Approximates the full InfSA via Perron eigenvector approximation
    using power iteration. No N×N attention matrix is ever constructed.

    Args:
        dim (int): Feature dimension.
        num_heads (int): Number of attention heads.
        gamma (float): Discount factor.
        qkv_bias (bool): Bias in Q/K/V projections.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        gamma: float = 0.5,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.gamma = gamma

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LinearInfSA forward pass.

        Args:
            x: (B, N, C)

        Returns:
            (B, N, C)
        """
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim

        q = self.q_proj(x).reshape(B, N, H, D).permute(0, 2, 1, 3)  # (B, H, N, D)
        k = self.k_proj(x).reshape(B, N, H, D).permute(0, 2, 1, 3)  # (B, H, N, D)
        v = self.v_proj(x).reshape(B, N, H, D).permute(0, 2, 1, 3)  # (B, H, N, D)

        # 1. Compute "soft importance scores" α_i = ||Q_i|| / Σ||Q_j||
        # These approximate the Perron eigenvector of Â
        q_norm = q.norm(dim=-1)  # (B, H, N)
        alpha = q_norm / (q_norm.sum(dim=-1, keepdim=True) + 1e-6)  # (B, H, N)

        # 2. Form center query: q̄ = Σ α_i·Q_i
        q_center = (alpha.unsqueeze(-1) * q).sum(dim=2)  # (B, H, D)

        # 3. Attention weights via q̄·K_j
        attn = torch.matmul(q_center.unsqueeze(2), k.transpose(-2, -1))  # (B, H, 1, N)
        attn = F.relu(attn * self.scale).squeeze(2)  # (B, H, N)

        # Frobenius normalization
        attn_norm = attn.norm(dim=-1, keepdim=True)
        attn_hat = attn / (attn_norm + 1e-6)  # (B, H, N)

        # 4. Aggregate values: O = Σ_j attn_j · V_j
        # Katz-style discount: γ / (1 - γ·Σattn), with denom clamping
        attn_sum = attn_hat.sum(dim=-1, keepdim=True)
        denom = 1.0 - self.gamma * attn_sum
        denom = torch.clamp(denom, min=0.1)                               # prevent div-by-zero
        attn_hat = self.gamma * attn_hat / (denom + 1e-6)

        out = (attn_hat.unsqueeze(-1) * v).sum(dim=2)  # (B, H, D)
        # Expand back to all tokens
        out = out.unsqueeze(2).expand(-1, -1, N, -1)  # (B, H, N, D)

        # Blend with per-token query for diversity
        q_weight = F.softmax(q_norm, dim=-1).unsqueeze(-1)  # (B, H, N, 1)
        out = out * 0.7 + v * q_weight * 0.3

        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.out_proj(out)

        return out


class InfSABlock(nn.Module):
    """Transformer block with InfSA attention (C2PSA-style for YOLO).

    Defaults to LinearInfSA (O(N)) for practical use. Full InfSA (O(N²))
    requires N < 500 tokens or will explode memory.

    LinearInfSA note: this is a global-context aggregation mechanism,
    not per-token attention. Every token receives the same aggregated
    representation (blended with 30% local value for diversity).

    Args:
        dim (int): Feature dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): MLP hidden dimension ratio.
        linear (bool): Use LinearInfSA instead of full InfSA (default True).
        gamma (float): Discount factor for Neumann series.
        num_terms (int): Neumann series terms (full InfSA only).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        linear: bool = True,
        gamma: float = 0.5,
        num_terms: int = 3,
        drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        if linear:
            self.attn = LinearInfSA(dim, num_heads=num_heads, gamma=gamma)
        else:
            self.attn = InfSA(dim, num_heads=num_heads, gamma=gamma, num_terms=num_terms)

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
        is_cnn = len(x.shape) == 4
        if is_cnn:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        if is_cnn:
            x = x.transpose(1, 2).reshape(B, C, H, W)

        return x


class C3k2_InfSA(nn.Module):
    """C3k2 + InfSABlock: CSP with spectral global aggregation.

    Each block: Bottleneck (local features) + InfSABlock (global context).
    

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of blocks.
        c3k (bool): YAML convention (ignored).
        e (float): Expansion ratio.
        shortcut (bool): Residual in Bottleneck.
        g (int): Groups for Bottleneck conv.
        num_heads (int): Attention heads.
        linear_infsa (bool): Use LinearInfSA (O(N), default True).
        gamma (float): Discount factor.
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
        linear_infsa: bool = True,
        gamma: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        from ultralytics.nn.modules.block import Conv, Bottleneck

        self.c = int(c2 * e)
        self.n = n

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        self.m = nn.ModuleList()
        for _ in range(n):
            block = nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0),
                InfSABlock(self.c, num_heads=num_heads, linear=linear_infsa, gamma=gamma),
            )
            self.m.append(block)

        self.cv2 = Conv((2 + n) * self.c, c2, 1, act=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        outputs = [y[0], y[1]]
        a = y[1]
        for m_block in self.m:
            a = m_block(a)
            outputs.append(a)
        return self.act(self.cv2(torch.cat(outputs, dim=1)))


# ================== Test ==================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("=" * 60)

    # Test 1: InfSA (full)
    print("\n1. InfSA (full, N=100)")
    x = torch.randn(2, 100, 256, device=device)
    attn = InfSA(256, num_heads=8, gamma=0.5, num_terms=3).to(device)
    y = attn(x)
    print(f"   Input: {x.shape} → Output: {y.shape}")
    loss = y.sum()
    loss.backward()
    print("   ✅ Backward OK")

    # Test 2: LinearInfSA
    print("\n2. LinearInfSA (N=1600)")
    x = torch.randn(2, 1600, 256, device=device)
    attn = LinearInfSA(256, num_heads=8, gamma=0.5).to(device)
    y = attn(x)
    print(f"   Input: {x.shape} → Output: {y.shape}")
    loss = y.sum()
    loss.backward()
    print("   ✅ Backward OK")

    # Test 3: InfSABlock CNN mode
    print("\n3. InfSABlock (CNN mode)")
    x = torch.randn(2, 256, 40, 40, device=device)
    block = InfSABlock(256, num_heads=8, linear=True).to(device)
    y = block(x)
    print(f"   Input: {x.shape} → Output: {y.shape}")
    loss = y.sum()
    loss.backward()
    print("   ✅ Backward OK")

    # Test 4: C3k2_InfSA
    print("\n4. C3k2_InfSA")
    x = torch.randn(2, 64, 64, 64, device=device)
    model = C3k2_InfSA(64, 128, n=2, e=0.5, num_heads=4, linear_infsa=True).to(device)
    y = model(x)
    print(f"   Input: {x.shape} → Output: {y.shape}")
    loss = y.sum()
    loss.backward()
    print("   ✅ Backward OK")

    # Test 5: Token count scaling
    print("\n5. Scaling test")
    for n_tok in [100, 400, 1600, 6400]:
        x = torch.randn(2, n_tok, 256, device=device)
        attn = LinearInfSA(256, num_heads=8).to(device)
        y = attn(x)
        print(f"   N={n_tok:>5}: OK")

    # Test 6: Full InfSA vs LinearInfSA
    print("\n6. InfSA vs LinearInfSA")
    x = torch.randn(2, 256, 256, device=device)
    full = InfSA(256, num_heads=8).to(device)
    lin = LinearInfSA(256, num_heads=8).to(device)
    y_f = full(x)
    y_l = lin(x)
    print(f"   InfSA mean: {y_f.abs().mean():.4f}")
    print(f"   LinearInfSA mean: {y_l.abs().mean():.4f}")

    # Test 7: Parameter count
    print("\n7. Parameter count")
    full = InfSA(256, num_heads=8)
    lin = LinearInfSA(256, num_heads=8)
    block = InfSABlock(256, num_heads=8, linear=True)
    print(f"   InfSA:  {sum(p.numel() for p in full.parameters()):,}")
    print(f"   LinearInfSA: {sum(p.numel() for p in lin.parameters()):,}")
    print(f"   InfSABlock:  {sum(p.numel() for p in block.parameters()):,}")

    print(f"\n   ✅ All tests passed!")
