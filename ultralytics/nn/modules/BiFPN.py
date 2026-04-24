import torch
import torch.nn as nn
import torch.nn.functional as F


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


class BiFPN(nn.Module):
    """
    支持原始 + 稀疏化特征平等融合的 BiFPN，自动对齐各层通道。

    参数:
        channels: 两种格式:
            - 列表: [[orig_c3, orig_c4, orig_c5], [sparse_c3, sparse_c4, sparse_c5]]
            - 单个列表: [c3, c4, c5] (向后兼容)
        out_channels: 输出通道 [o3, o4, o5]，默认 o3=c3//2, o4=c4, o5=c5
    """
    def __init__(self, channels, out_channels=None, use_depthwise=True):
        super().__init__()
        # 解析参数
        if isinstance(channels[0], (list, tuple)):
            orig_c, sparse_c = channels
        else:
            orig_c = sparse_c = channels

        c3_orig, c4_orig, c5_orig = orig_c
        c3_sparse, c4_sparse, c5_sparse = sparse_c

        if out_channels is None:
            o3 = c3_orig // 2
            o4 = c4_orig
            o5 = c5_orig
        else:
            o3, o4, o5 = out_channels

        # ---- 原始特征投影 ----
        self.p3_orig_proj = nn.Conv2d(c3_orig, o3, 1, bias=False)
        self.p4_orig_proj = nn.Conv2d(c4_orig, o4, 1, bias=False)
        self.p5_orig_proj = nn.Conv2d(c5_orig, o5, 1, bias=False)

        # ---- 稀疏特征投影 ----
        self.p3_sparse_proj = nn.Conv2d(c3_sparse, o3, 1, bias=False)
        self.p4_sparse_proj = nn.Conv2d(c4_sparse, o4, 1, bias=False)
        self.p5_sparse_proj = nn.Conv2d(c5_sparse, o5, 1, bias=False)

        # ---- 通道对齐用投影（跨尺度时使用）----
        self.p5_to_p4 = nn.Conv2d(o5, o4, 1, bias=False)  # 256 → 128
        self.p4_to_p3 = nn.Conv2d(o4, o3, 1, bias=False)  # 128 → 64 (如果o3=c3//2)
        self.p3_to_p4 = nn.Conv2d(o3, o4, 1, bias=False)  # 64  → 128
        self.p4_to_p5 = nn.Conv2d(o4, o5, 1, bias=False)  # 128 → 256

        # ---- 融合节点 ----
        self.p5_fuse = BiFPN_Add(2)
        self.p4_fuse = BiFPN_Add(3)
        self.p3_fuse = BiFPN_Add(3)
        self.p4_out_fuse = BiFPN_Add(2)
        self.p5_out_fuse = BiFPN_Add(2)

        # ---- 平滑卷积 ----
        ConvLayer = lambda c: nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False) if use_depthwise \
                       else nn.Conv2d(c, c, 3, padding=1, bias=False)

        self.smooth_p5 = ConvLayer(o5)
        self.smooth_p4_td = ConvLayer(o4)
        self.smooth_p3_td = ConvLayer(o3)
        self.smooth_p4_out = ConvLayer(o4)
        self.smooth_p5_out = ConvLayer(o5)

    def forward(self, features):
        p3_orig, p4_orig, p5_orig, p3_sparse, p4_sparse, p5_sparse = features

        # 投影到各自输出通道
        p3_orig = self.p3_orig_proj(p3_orig)      # (B, o3, H/8, W/8)
        p4_orig = self.p4_orig_proj(p4_orig)      # (B, o4, H/16, W/16)
        p5_orig = self.p5_orig_proj(p5_orig)      # (B, o5, H/32, W/32)
        p3_sparse = self.p3_sparse_proj(p3_sparse)
        p4_sparse = self.p4_sparse_proj(p4_sparse)
        p5_sparse = self.p5_sparse_proj(p5_sparse)

        # ---- 自顶向下 ----
        # P5 融合 (只含原始+稀疏，仍在 P5 尺度)
        p5_td = self.smooth_p5(self.p5_fuse([p5_orig, p5_sparse]))  # (B, o5, H/32, W/32)

        # P4 融合：原始 + 稀疏 + P5 上采样，需要把 P5 通道从 o5 降到 o4
        p5_up = F.interpolate(p5_td, size=p4_orig.shape[-2:], mode='nearest')  # (B, o5, H/16, W/16)
        p5_up_aligned = self.p5_to_p4(p5_up)  # (B, o4, H/16, W/16)
        p4_td = self.smooth_p4_td(self.p4_fuse([p4_orig, p4_sparse, p5_up_aligned]))

        # P3 融合：原始 + 稀疏 + P4 上采样，P4 → o3
        p4_up = F.interpolate(p4_td, size=p3_orig.shape[-2:], mode='nearest')  # (B, o4, H/8, W/8)
        p4_up_aligned = self.p4_to_p3(p4_up)  # (B, o3, H/8, W/8)
        p3_td = self.smooth_p3_td(self.p3_fuse([p3_orig, p3_sparse, p4_up_aligned]))

        # ---- 自底向上 ----
        # P4 输出：P4_td + P3 下采样，P3 → o4
        p3_down = F.avg_pool2d(p3_td, kernel_size=2, stride=2)  # (B, o3, H/16, W/16)
        p3_down_aligned = self.p3_to_p4(p3_down)                # (B, o4, H/16, W/16)
        p4_out = self.smooth_p4_out(self.p4_out_fuse([p4_td, p3_down_aligned]))

        # P5 输出：P5_td + P4_out 下采样，P4 → o5
        p4_down = F.avg_pool2d(p4_out, kernel_size=2, stride=2)  # (B, o4, H/32, W/32)
        p4_down_aligned = self.p4_to_p5(p4_down)                # (B, o5, H/32, W/32)
        p5_out = self.smooth_p5_out(self.p5_out_fuse([p5_td, p4_down_aligned]))

        return [p3_td, p4_out, p5_out]