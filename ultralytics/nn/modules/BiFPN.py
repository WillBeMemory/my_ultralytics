import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv

class BiFPN_Add(nn.Module):
    """快速归一化加权融合，支持自定义初始权重"""
    def __init__(self, num_inputs=2, init_weights=None):
        super().__init__()
        if init_weights is None:
            self.w = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        else:
            self.w = nn.Parameter(torch.tensor(init_weights, dtype=torch.float32))
        self.eps = 1e-4

    def forward(self, xs):
        w = F.relu(self.w)
        w_norm = w / (w.sum() + self.eps)
        return sum(w_norm[i] * xs[i] for i in range(len(xs)))

class BiFPN(nn.Module):
    """
    BiFPN 完整实现 + P4稀疏特征增强
    参数:
        channels: 输入通道列表 [P3, P4, P5]
        out_channels: 输出通道列表 (默认P3减半，P4/P5不变)
        p4_boost_init: P4增强因子的初始值 (默认1.5，使P4贡献更高)
    """
    def __init__(self, channels, out_channels=None, use_depthwise=True, p4_boost_init=1.5):
        super().__init__()
        c3, c4, c5 = channels
        if out_channels is None:
            o3, o4, o5 = c3 // 2, c4, c5
        else:
            o3, o4, o5 = out_channels

        # ---- P4 增强因子（可学习） ----
        self.p4_boost = nn.Parameter(torch.tensor(p4_boost_init, dtype=torch.float32))

        # ---- 通道投影 ----
        self.p3_proj = nn.Conv2d(c3, o3, 1, bias=False)
        self.p5_to_p4 = nn.Conv2d(c5, o4, 1, bias=False)
        self.p4_to_p3 = nn.Conv2d(o4, o3, 1, bias=False)
        self.p3_to_p4 = nn.Conv2d(o3, o4, 1, bias=False)
        self.p4_to_p5 = nn.Conv2d(o4, o5, 1, bias=False)

        # ---- 加权融合节点（P4分支初始权重更高） ----
        self.p4_td_fuse = BiFPN_Add(2, init_weights=[1.5, 1.0])  # [P4权重, P5_up权重]
        self.p3_td_fuse = BiFPN_Add(2, init_weights=[1.0, 1.0])  # [P3权重, P4_td_up权重]
        self.p4_bu_fuse = BiFPN_Add(2, init_weights=[1.5, 1.0])  # [P4_td权重, P3_down权重]
        self.p5_bu_fuse = BiFPN_Add(2, init_weights=[1.0, 1.0])  # [P5权重, P4_down权重]

        # ---- 平滑卷积 ----
        if use_depthwise:
            self.smooth_p4_td = nn.Conv2d(o4, o4, 3, padding=1, groups=o4, bias=False)
            self.smooth_p3_td = nn.Conv2d(o3, o3, 3, padding=1, groups=o3, bias=False)
            self.smooth_p4_out = nn.Conv2d(o4, o4, 3, padding=1, groups=o4, bias=False)
            self.smooth_p5_out = nn.Conv2d(o5, o5, 3, padding=1, groups=o5, bias=False)
        else:
            self.smooth_p4_td = nn.Conv2d(o4, o4, 3, padding=1, bias=False)
            self.smooth_p3_td = nn.Conv2d(o3, o3, 3, padding=1, bias=False)
            self.smooth_p4_out = nn.Conv2d(o4, o4, 3, padding=1, bias=False)
            self.smooth_p5_out = nn.Conv2d(o5, o5, 3, padding=1, bias=False)

    def forward(self, features):
        p3, p4, p5 = features

        # ---- P4 特征增强 ----
        p4 = p4 * self.p4_boost

        # ---- 自顶向下 ----
        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        p4_td = self.smooth_p4_td(self.p4_td_fuse([p4, self.p5_to_p4(p5_up)]))

        p4_td_up = F.interpolate(p4_td, size=p3.shape[-2:], mode='nearest')
        p3_td = self.smooth_p3_td(self.p3_td_fuse([self.p3_proj(p3), self.p4_to_p3(p4_td_up)]))

        # ---- 自底向上 ----
        p3_down = F.avg_pool2d(p3_td, kernel_size=2, stride=2)
        p4_out = self.smooth_p4_out(self.p4_bu_fuse([p4_td, self.p3_to_p4(p3_down)]))

        p4_down = F.avg_pool2d(p4_out, kernel_size=2, stride=2)
        p5_out = self.smooth_p5_out(self.p5_bu_fuse([p5, self.p4_to_p5(p4_down)]))

        return [p3_td, p4_out, p5_out]