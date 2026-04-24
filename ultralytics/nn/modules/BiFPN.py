# ============================================
# File: BiFPN_with_GSConv.py
# 依赖：ultralytics.nn.modules.GSConv 中的 GSConv
# 功能：融合了 GSConv 选项的 BiFPN，支持原始+稀疏特征融合
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.GSConv import GSConv   # 导入已有的 GSConv


# ------------------------------------------------------------
# 1. 快速归一化加权融合
# ------------------------------------------------------------
class BiFPN_Add(nn.Module):
    """快速归一化加权融合，支持任意数量的输入"""
    def __init__(self, num_inputs=2):
        super().__init__()
        self.w = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.eps = 1e-4

    def forward(self, xs):
        w = F.relu(self.w)
        w_norm = w / (w.sum() + self.eps)
        return sum(w_norm[i] * xs[i] for i in range(len(xs)))


# ------------------------------------------------------------
# 2. BiFPN 主体（融合 GSConv 选项）
# ------------------------------------------------------------
class BiFPN(nn.Module):
    """
    支持原始 + 稀疏化特征平等融合的 BiFPN，可选 GSConv 平滑

    参数:
        channels: 通道配置，两种格式:
            - 列表: [[orig_c3, orig_c4, orig_c5], [sparse_c3, sparse_c4, sparse_c5]]
            - 单个列表: [c3, c4, c5] (向后兼容，此时原始与稀疏通道相同)
        out_channels: 输出通道 [o3, o4, o5]，默认 o3=c3//2, o4=c4, o5=c5
        use_depthwise: 若 use_gsconv=False，则平滑层使用深度可分离卷积（groups=cin）
        use_gsconv: 是否使用 GSConv 作为平滑层（优先级高于 use_depthwise）
    """
    def __init__(self, channels, out_channels=None, use_depthwise=False, use_gsconv=True):
        super().__init__()
        # 解析输入通道
        if isinstance(channels[0], (list, tuple)):
            orig_c, sparse_c = channels  # 原始特征通道，稀疏特征通道
        else:
            orig_c = sparse_c = channels

        c3_orig, c4_orig, c5_orig = orig_c
        c3_sparse, c4_sparse, c5_sparse = sparse_c

        # 确定输出通道
        if out_channels is None:
            o3 = c3_orig // 2
            o4 = c4_orig
            o5 = c5_orig
        else:
            o3, o4, o5 = out_channels

        # ---- 原始特征投影（1x1）----
        self.p3_orig_proj = nn.Conv2d(c3_orig, o3, 1, bias=False)
        self.p4_orig_proj = nn.Conv2d(c4_orig, o4, 1, bias=False)
        self.p5_orig_proj = nn.Conv2d(c5_orig, o5, 1, bias=False)

        # ---- 稀疏特征投影（1x1）----
        self.p3_sparse_proj = nn.Conv2d(c3_sparse, o3, 1, bias=False)
        self.p4_sparse_proj = nn.Conv2d(c4_sparse, o4, 1, bias=False)
        self.p5_sparse_proj = nn.Conv2d(c5_sparse, o5, 1, bias=False)

        # ---- 跨尺度通道对齐（1x1）----
        self.p5_to_p4 = nn.Conv2d(o5, o4, 1, bias=False)
        self.p4_to_p3 = nn.Conv2d(o4, o3, 1, bias=False)
        self.p3_to_p4 = nn.Conv2d(o3, o4, 1, bias=False)
        self.p4_to_p5 = nn.Conv2d(o4, o5, 1, bias=False)

        # ---- 融合节点 ----
        self.p5_fuse = BiFPN_Add(2)
        self.p4_fuse = BiFPN_Add(3)
        self.p3_fuse = BiFPN_Add(3)
        self.p4_out_fuse = BiFPN_Add(2)
        self.p5_out_fuse = BiFPN_Add(2)

        # ---- 平滑层：根据标志选择 ----
        if use_gsconv:
            # 使用 GSConv，保持通道数不变，步长1，激活SiLU
            smooth_fn = lambda c: GSConv(c, c, k=3, s=1, act=True)
        elif use_depthwise:
            # 深度可分离卷积（groups = c）
            smooth_fn = lambda c: nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        else:
            # 标准 3x3 卷积
            smooth_fn = lambda c: nn.Conv2d(c, c, 3, padding=1, bias=False)

        self.smooth_p5 = smooth_fn(o5)
        self.smooth_p4_td = smooth_fn(o4)
        self.smooth_p3_td = smooth_fn(o3)
        self.smooth_p4_out = smooth_fn(o4)
        self.smooth_p5_out = smooth_fn(o5)

    def forward(self, features):
        # 解包6个输入特征
        p3_orig, p4_orig, p5_orig, p3_sparse, p4_sparse, p5_sparse = features

        # 1. 投影到目标通道
        p3_orig = self.p3_orig_proj(p3_orig)
        p4_orig = self.p4_orig_proj(p4_orig)
        p5_orig = self.p5_orig_proj(p5_orig)
        p3_sparse = self.p3_sparse_proj(p3_sparse)
        p4_sparse = self.p4_sparse_proj(p4_sparse)
        p5_sparse = self.p5_sparse_proj(p5_sparse)

        # ---- 自顶向下 ----
        p5_td = self.smooth_p5(self.p5_fuse([p5_orig, p5_sparse]))          # P5

        p5_up = F.interpolate(p5_td, size=p4_orig.shape[-2:], mode='nearest')
        p5_up_aligned = self.p5_to_p4(p5_up)                                # o5 -> o4
        p4_td = self.smooth_p4_td(self.p4_fuse([p4_orig, p4_sparse, p5_up_aligned]))

        p4_up = F.interpolate(p4_td, size=p3_orig.shape[-2:], mode='nearest')
        p4_up_aligned = self.p4_to_p3(p4_up)                                # o4 -> o3
        p3_td = self.smooth_p3_td(self.p3_fuse([p3_orig, p3_sparse, p4_up_aligned]))

        # ---- 自底向上 ----
        p3_down = F.avg_pool2d(p3_td, kernel_size=2, stride=2)
        p3_down_aligned = self.p3_to_p4(p3_down)                            # o3 -> o4
        p4_out = self.smooth_p4_out(self.p4_out_fuse([p4_td, p3_down_aligned]))

        p4_down = F.avg_pool2d(p4_out, kernel_size=2, stride=2)
        p4_down_aligned = self.p4_to_p5(p4_down)                            # o4 -> o5
        p5_out = self.smooth_p5_out(self.p5_out_fuse([p5_td, p4_down_aligned]))

        return [p3_td, p4_out, p5_out]


# ------------------------------------------------------------
# 3. 测试 main
# ------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 模拟 YOLO11n 缩放后 (width=0.25) 的实际通道
    orig_channels = [128, 128, 256]      # P3, P4, P5 原始特征通道
    sparse_channels = [128, 128, 256]    # HIPA 输出通道

    # 构建 BiFPN，启用 GSConv
    bifpn = BiFPN(
        channels=[orig_channels, sparse_channels],
        out_channels=None,           # 自动 o3=64, o4=128, o5=256
        use_gsconv=True
    ).to(device)

    # 构造假输入 (batch=2, 分辨率 640x640)
    inp = [
        torch.randn(2, 128, 80, 80).to(device),   # p3_orig
        torch.randn(2, 128, 40, 40).to(device),   # p4_orig
        torch.randn(2, 256, 20, 20).to(device),   # p5_orig
        torch.randn(2, 128, 80, 80).to(device),   # p3_sparse
        torch.randn(2, 128, 40, 40).to(device),   # p4_sparse
        torch.randn(2, 256, 20, 20).to(device),   # p5_sparse
    ]

    # 前向
    out = bifpn(inp)

    # 检查输出形状
    print("\n=== BiFPN + GSConv 输出形状 ===")
    for i, name in enumerate(["P3_out", "P4_out", "P5_out"]):
        print(f"{name}: {out[i].shape}")

    expected = [(2, 64, 80, 80), (2, 128, 40, 40), (2, 256, 20, 20)]
    for i, (o, e) in enumerate(zip(out, expected)):
        assert o.shape == e, f"Shape mismatch at output {i}: expected {e}, got {o.shape}"
    print("所有输出形状验证通过！")

    # 参数量统计
    total_params = sum(p.numel() for p in bifpn.parameters())
    trainable_params = sum(p.numel() for p in bifpn.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")