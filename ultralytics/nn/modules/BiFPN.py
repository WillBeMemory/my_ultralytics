# ============================================
# File: ultralytics/nn/modules/BiFPN.py
# 新版 BiFPN：支持原始双向（6 输入）与纯自顶向下（3 输入）
# 当输入特征数量为 3 时，自动切换为 Top‑Down 模式，仅由 P5 引导融合
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.GSConv import GSConv   # 可选


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
    自适应 BiFPN：
    - 若传入 channels 为嵌套列表（[[c3o,c4o,c5o],[c3s,c4s,c5s]]），
      则按照原始双向融合，输入需为 6 个特征图（原始+稀疏）。
    - 若传入 channels 为普通列表（[c3, c4, c5]），
      则执行纯自顶向下融合，输入只需 3 个特征图（P3, P4, P5）。
    """
    def __init__(self, channels, out_channels=None, use_depthwise=False, use_gsconv=False):
        super().__init__()
        # ----- 判断是否为双流模式 -----
        self.dual_stream = isinstance(channels[0], (list, tuple))

        if self.dual_stream:
            orig_c, sparse_c = channels
            c3_orig, c4_orig, c5_orig = orig_c
            c3_sparse, c4_sparse, c5_sparse = sparse_c
            if out_channels is None:
                o3 = c3_orig // 2
                o4 = c4_orig
                o5 = c5_orig
            else:
                o3, o4, o5 = out_channels

            # 原始特征投影
            self.p3_orig_proj = nn.Conv2d(c3_orig, o3, 1, bias=False)
            self.p4_orig_proj = nn.Conv2d(c4_orig, o4, 1, bias=False)
            self.p5_orig_proj = nn.Conv2d(c5_orig, o5, 1, bias=False)
            # 稀疏特征投影
            self.p3_sparse_proj = nn.Conv2d(c3_sparse, o3, 1, bias=False)
            self.p4_sparse_proj = nn.Conv2d(c4_sparse, o4, 1, bias=False)
            self.p5_sparse_proj = nn.Conv2d(c5_sparse, o5, 1, bias=False)
            # 通道对齐（双向）
            self.p5_to_p4 = nn.Conv2d(o5, o4, 1, bias=False)
            self.p4_to_p3 = nn.Conv2d(o4, o3, 1, bias=False)
            self.p3_to_p4 = nn.Conv2d(o3, o4, 1, bias=False)
            self.p4_to_p5 = nn.Conv2d(o4, o5, 1, bias=False)
            # 融合节点
            self.p5_fuse = BiFPN_Add(2)
            self.p4_fuse = BiFPN_Add(3)
            self.p3_fuse = BiFPN_Add(3)
            self.p4_out_fuse = BiFPN_Add(2)
            self.p5_out_fuse = BiFPN_Add(2)
            # 平滑卷积
            smooth_fn = self._get_smooth_fn(use_gsconv, use_depthwise)
            self.smooth_p5 = smooth_fn(o5)
            self.smooth_p4_td = smooth_fn(o4)
            self.smooth_p3_td = smooth_fn(o3)
            self.smooth_p4_out = smooth_fn(o4)
            self.smooth_p5_out = smooth_fn(o5)

        else:
            # ========== 纯 Top‑Down 模式 ==========
            c3, c4, c5 = channels
            if out_channels is None:
                o3 = c3 // 2
                o4 = c4
                o5 = c5
            else:
                o3, o4, o5 = out_channels

            # 投影层
            self.p3_proj = nn.Conv2d(c3, o3, 1, bias=False)
            self.p4_proj = nn.Conv2d(c4, o4, 1, bias=False)
            self.p5_proj = nn.Conv2d(c5, o5, 1, bias=False)

            # 通道对齐（P5→P4, P4→P3）
            self.p5_to_p4 = nn.Conv2d(o5, o4, 1, bias=False)
            self.p4_to_p3 = nn.Conv2d(o4, o3, 1, bias=False)

            # 融合节点（每个只有两个输入）
            self.p4_fuse = BiFPN_Add(2)
            self.p3_fuse = BiFPN_Add(2)

            # 平滑卷积（P5 直接输出，不加平滑）
            smooth_fn = self._get_smooth_fn(use_gsconv, use_depthwise)
            self.smooth_p4 = smooth_fn(o4)
            self.smooth_p3 = smooth_fn(o3)

    @staticmethod
    def _get_smooth_fn(use_gsconv, use_depthwise):
        if use_gsconv:
            return lambda c: GSConv(c, c, k=3, s=1, act=True)
        elif use_depthwise:
            return lambda c: nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        else:
            return lambda c: nn.Conv2d(c, c, 3, padding=1, bias=False)

    def forward(self, features):
        if self.dual_stream:
            # 原始双向逻辑（保持不变，省略，用你已有的实现）
            # 这里仅示意，实际应使用你之前的多输入融合代码
            raise NotImplementedError("Dual-stream forward not shown for brevity.")
        else:
            # ========== Top‑Down 前向 ==========
            p3, p4, p5 = features

            p3 = self.p3_proj(p3)
            p4 = self.p4_proj(p4)
            p5 = self.p5_proj(p5)          # P5 投影后直接作为输出

            # P4 融合：P4_orig + P5 上采样
            p5_up = F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
            p5_up = self.p5_to_p4(p5_up)   # 对齐通道
            p4_td = self.smooth_p4(self.p4_fuse([p4, p5_up]))

            # P3 融合：P3_orig + P4_td 上采样
            p4_up = F.interpolate(p4_td, size=p3.shape[-2:], mode='nearest')
            p4_up = self.p4_to_p3(p4_up)   # 对齐通道
            p3_td = self.smooth_p3(self.p3_fuse([p3, p4_up]))

            return [p3_td, p4_td, p5]      # 输出顺序：P3, P4, P5


# ============================================
# 测试代码（自顶向下模式）
# ============================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 模拟 YOLO11n 缩放后通道 (width=0.25)
    # 原始配置：[512, 512, 1024] → 缩放后 [128, 128, 256]
    channels = [128, 128, 256]   # 对应 P3, P4, P5 实际输入通道

    model = BiFPN(channels, use_depthwise=True, use_gsconv=False).to(device)

    # 创建假输入
    p3 = torch.randn(2, 128, 80, 80).to(device)
    p4 = torch.randn(2, 128, 40, 40).to(device)
    p5 = torch.randn(2, 256, 20, 20).to(device)

    out = model([p3, p4, p5])
    print("=== Top‑Down BiFPN Output Shapes ===")
    for i, name in enumerate(["P3", "P4", "P5"]):
        print(f"{name}: {out[i].shape}")
    # 期望：P3 (2,64,80,80), P4 (2,128,40,40), P5 (2,256,20,20)

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}")