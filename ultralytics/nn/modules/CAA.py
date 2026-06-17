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
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0)
        self.act = nn.GELU()
        self.conv_h = nn.Conv2d(ch, ch, kernel_size=(1, h_kernel_size), stride=1,
                                padding=(0, h_kernel_size // 2), groups=ch)
        self.conv_v = nn.Conv2d(ch, ch, kernel_size=(v_kernel_size, 1), stride=1,
                                padding=(v_kernel_size // 2, 0), groups=ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

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


class CAABottleneck(Bottleneck):
    """Bottleneck + CAA：在第二个 conv 之后接 CAA 注意力（对标 LSKBottleneck）。"""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=1.0,
                 h_kernel_size=11, v_kernel_size=11):
        super().__init__(c1, c2, shortcut, g, k, e)
        self.caa = CAA(c2, h_kernel_size, v_kernel_size)

    def forward(self, x):
        identity = x
        out = self.cv2(self.cv1(x))
        out = self.caa(out)
        return identity + out if self.add else out


class C3k2_CAA(C2f):
    """C3k2 with CAA attention in bottlenecks（对标 C3k2_LSK）。

    继承 C2f，将每个分支的 Bottleneck 替换为 CAABottleneck（c3k=True 时则用 C3k）。
    yaml 用法与 C3k2 完全一致，仅模块名不同：
        [-1, 2, C3k2_CAA, [256, False, 0.25]]
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True,
                 h_kernel_size=11, v_kernel_size=11):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else
            CAABottleneck(self.c, self.c, shortcut, g, e=1.0,
                          h_kernel_size=h_kernel_size, v_kernel_size=v_kernel_size)
            for _ in range(n)
        )


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
