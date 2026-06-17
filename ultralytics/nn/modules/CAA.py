import torch
import torch.nn as nn
from ultralytics.nn.modules.block import C2f, Bottleneck, C3k


class CAA(nn.Module):
    """Context Anchor Attention（PKINet, CVPR 2024）。

    用「条状深度卷积」近似大核深度卷积，捕获长程上下文——对各向异性/细长目标
    （如 SAR 船舶）的朝向特征尤为有效。

    数据流（对齐 PKINet 官方实现）：
        AvgPool(k=7,s=1,p=3) → 1×1 → GELU → DW conv_h(1×k) → DW conv_v(k×1)
        → GELU → 1×1 → Sigmoid → 与输入逐元素相乘(门控)

    说明：
      - conv_h / conv_v 为 depthwise（groups=ch），忠实 PKINet「用 1D 条状深度卷积
        近似 2D 大核深度卷积」的设计动机（参数 O(2k·C) 近似 O(k²·C) 感受野）；
      - AvgPool 必须 stride=1,padding=3 才能保持空间尺寸，以支持与输入的 element-wise 门控；
      - h/v_kernel_size 默认 11（PKINet 默认值）。
    """

    def __init__(self, ch, h_kernel_size=11, v_kernel_size=11):
        super().__init__()
        # --- RNG 解耦：保存 → 构建 → 恢复，使 CAA 对全局 torch RNG 净消耗为 0 ---
        # 关键：Conv2d 在构造时即调用 reset_parameters()→kaiming_uniform_，消耗全局 RNG。
        # 若不隔离，插入 CAA 会让排在它之后的层（同一 backbone/neck 里后构建的层）初始权重
        # 序列整体后移，与「模块前向作用」混在一起，无法干净归因 ΔmAP（seed coupling）。
        # save/restore 后，CAA 之后的层初值与「未插 CAA 的基线」逐比特一致。
        # 这是 PyTorch 官方 reproducibility 文档推荐的隔离模式，且不影响跨 seed 的整体随机性。
        _cpu_rng = torch.get_rng_state()
        _cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0)
        self.act = nn.GELU()
        self.conv_h = nn.Conv2d(ch, ch, kernel_size=(1, h_kernel_size), stride=1,
                                padding=(0, h_kernel_size // 2), groups=ch)
        self.conv_v = nn.Conv2d(ch, ch, kernel_size=(v_kernel_size, 1), stride=1,
                                padding=(v_kernel_size // 2, 0), groups=ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        # Identity 初始化：conv2 权重置零 + 正偏置，使初始 sigmoid(bias) ≈ 1（≈ identity）。
        # 否则随机初始化下初始 attn ≈ 0.5，会把特征整体缩放/不均匀加权，扰乱 backbone
        # 输出，导致前若干 epoch 性能先变差再恢复（同理于 Pixel_BiFPN_Add 的零初始化）。
        nn.init.zeros_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 4.0)   # sigmoid(4) ≈ 0.982 ≈ 1

        # 恢复全局 RNG —— CAA 之后的层初始化序列回到「没插 CAA」时的原位
        torch.set_rng_state(_cpu_rng)
        if _cuda_rng is not None:
            torch.cuda.set_rng_state(_cuda_rng)

    def forward(self, x):
        attn = self.avg_pool(x)
        attn = self.conv1(attn)
        attn = self.act(attn)
        attn = self.conv_h(attn)
        attn = self.conv_v(attn)
        attn = self.act(attn)
        attn = self.conv2(attn)
        attn = self.sigmoid(attn)
        return attn * x


class C3k2_CAA(C2f):
    """C3k2 + 输出级 CAA 注意力（忠实 PKINet 的 stage-attention 定位）。

    设计依据：CAA 在 PKINet 中是 block 级注意力，接在主卷积处理之后对整块输出
    做门控，而非嵌进每个内部 bottleneck。因此本类保持标准 C3k2 的 CSP 结构
    （Bottleneck 链 + cv2）完全不变，仅在最终输出 (cv2 之后) 接一个 CAA。
    每个 C3k2_CAA 只含 1 个 CAA（与 n 无关），且 Bottleneck 用标准 e=0.5 不膨胀。

    yaml 用法与 C3k2 一致：[-1, 2, C3k2_CAA, [256, False, 0.25]]
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True,
                 h_kernel_size=11, v_kernel_size=11):
        super().__init__(c1, c2, n, shortcut, g, e)
        # 标准 C3k2 分支构建（c3k=True 用 C3k，否则标准 Bottleneck e=0.5），不改动
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else
            Bottleneck(self.c, self.c, shortcut, g, e=0.5)
            for _ in range(n)
        )
        # CAA 作为 stage-level 注意力，门控 cv2 输出（c2 通道）
        self.caa = CAA(c2, h_kernel_size, v_kernel_size)

    def forward(self, x):
        return self.caa(super().forward(x))


# ================== 测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- CAA 单算子 ---
    print("\n=== CAA 单算子 ===")
    caa = CAA(64).to(device)
    x = torch.randn(2, 64, 32, 32, device=device)
    y = caa(x)
    assert y.shape == x.shape, f"CAA 应保持空间尺寸: {y.shape} vs {x.shape}"
    # 手动复算注意力权重，验证 ∈[0,1]
    with torch.no_grad():
        attn = caa.sigmoid(caa.conv2(caa.act(caa.conv_v(caa.conv_h(caa.act(caa.conv1(caa.avg_pool(x))))))))
    print(f"  in/out: {tuple(x.shape)} -> {tuple(y.shape)} (保持尺寸 OK)")
    print(f"  注意力范围: attn.min={attn.min().item():.4f} max={attn.max().item():.4f} (应 ∈[0,1])")
    print(f"  CAA params: {sum(p.numel() for p in caa.parameters()):,}")

    # --- C3k2_CAA vs C3k2_LSK 对比 ---
    from ultralytics.nn.modules.lsknet import C3k2_LSK
    c1, c2 = 128, 128
    for name, cls in [('C3k2_CAA', C3k2_CAA), ('C3k2_LSK', C3k2_LSK)]:
        m = cls(c1, c2, n=2).to(device)
        xin = torch.randn(2, c1, 16, 16, device=device)
        out = m(xin)
        assert out.shape == xin.shape, f"{name} shape: {out.shape}"
        loss = out.sum()
        loss.backward()
        n_caa = sum(1 for mo in m.modules() if isinstance(mo, CAA))
        print(f"  {name:10s}: out {tuple(out.shape)} | params {sum(p.numel() for p in m.parameters()):,} | CAA实例={n_caa}")
    print("\nAll checks passed.")
