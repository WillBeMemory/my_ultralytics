import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import C3k2, Conv
from ultralytics.nn.modules import HWD
from ultralytics.nn.modules.WaveletFusionUp import WaveletFusionUp


class BiFPN_Add(nn.Module):
    """快速归一化加权融合，支持任意数量的输入。"""
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
    双向特征金字塔网络（EfficientDet 风格），固定三尺度输出。
    当输入包含 P2 时，将其下采样并融合到 P3（可选）。

    参数:
        channels       : 输入通道列表，长度 3 (P3,P4,P5) 或 4 (P2,P3,P4,P5)
        out_channels   : 输出通道列表，长度必须为 3，默认使用 P3-P5 通道。
        fuse_p2        : 是否启用 P2 融合（仅当长度=4 时生效）
        p2_to_p3_down  : 下采样 P2 的方式 "hwd" 或 "conv"
    """
    def __init__(
        self,
        channels,
        out_channels=None,
        use_wavelet_up=False,
        use_hwd_down=True,
        use_c3k2=False,
        c3k2_kwargs=None,
        fuse_p2=True,
        p2_to_p3_down="hwd",
    ):
        super().__init__()
        self.num_inputs = len(channels)
        self.fuse_p2 = fuse_p2 and (self.num_inputs == 4)

        # 确定 P3、P4、P5 通道
        if self.num_inputs == 4:
            c2, c3, c4, c5 = channels
        else:
            c3, c4, c5 = channels

        self.c3, self.c4, self.c5 = c3, c4, c5

        if out_channels is None:
            out_channels = [c3, c4, c5]
        self.out_channels = out_channels

        kwa = c3k2_kwargs if c3k2_kwargs else dict(n=1, shortcut=False, e=0.5)

        # ---------- P2 融合模块 ----------
        if self.fuse_p2:
            if p2_to_p3_down == "hwd":
                self.p2_down = HWD(c2, c3)
            else:
                self.p2_down = Conv(c2, c3, k=3, s=2, act=True)
        else:
            self.p2_down = None

        # ---------- 通道对齐（1x1 卷积） ----------
        # 自顶向下通路：两个横向连接，顺序为 (P5->P4, P4->P3)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c5, c4, 1, bias=False),  # P5 -> P4
            nn.Conv2d(c4, c3, 1, bias=False),  # P4 -> P3
        ])
        # 小波上采样后的后置投影（仅在 use_wavelet_up=True 时使用）
        if use_wavelet_up:
            self.up_proj = nn.ModuleList([
                nn.Conv2d(c5, c4, 1, bias=False),
                nn.Conv2d(c4, c3, 1, bias=False),
            ])
        else:
            self.up_proj = nn.ModuleList([nn.Identity(), nn.Identity()])

        # 自底向上通路：两个下采样通道映射 (P3->P4, P4->P5)
        self.down_convs = nn.ModuleList([
            nn.Conv2d(c3, c4, 1, bias=False),
            nn.Conv2d(c4, c5, 1, bias=False),
        ])

        # ---------- 融合节点 ----------
        self.td_nodes = nn.ModuleList([BiFPN_Add(2), BiFPN_Add(2)])  # P5->P4, P4->P3
        self.bu_nodes = nn.ModuleList([BiFPN_Add(2), BiFPN_Add(2)])  # P3->P4, P4->P5

        # ---------- 上采样模块 ----------
        self.ups = nn.ModuleList()
        for i in range(2):
            if use_wavelet_up:
                low_ch = c4 if i == 0 else c3
                high_ch = c5 if i == 0 else c4
                self.ups.append(WaveletFusionUp(low_ch, high_ch, k=3))
            else:
                self.ups.append(None)

        # ---------- 下采样模块 ----------
        if use_hwd_down:
            self.downs = nn.ModuleList([HWD(c3, c4), HWD(c4, c5)])
        else:
            self.downs = nn.ModuleList([Conv(c3, c4, k=3, s=2, act=True),
                                        Conv(c4, c5, k=3, s=2, act=True)])

        # ---------- 可选后处理 ----------
        if use_c3k2:
            self.refines_td = nn.ModuleList([
                C3k2(out_channels[2], out_channels[2], **kwa),
                C3k2(out_channels[1], out_channels[1], **kwa),
                C3k2(out_channels[0], out_channels[0], **kwa),
            ])
            self.refines_bu = nn.ModuleList([
                C3k2(out_channels[0], out_channels[0], **kwa),
                C3k2(out_channels[1], out_channels[1], **kwa),
                C3k2(out_channels[2], out_channels[2], **kwa),
            ])
        else:
            self.refines_td = None
            self.refines_bu = None

    def forward(self, features):
        # 1. 处理 P2 融合
        if self.fuse_p2 and len(features) == 4:
            p2, p3, p4, p5 = features
            p2_down = self.p2_down(p2)
            p3 = p3 + p2_down
            features = [p3, p4, p5]
        elif len(features) == 4:
            features = features[1:]  # 丢弃 P2

        # 现在 features = [p3, p4, p5]
        p3, p4, p5 = features

        # 反转顺序，方便从高分辨率开始处理：P5, P4, P3
        feats = [p5, p4, p3]

        # ---------- 自顶向下 ----------
        td = [None, None, None]
        td[0] = feats[0]  # P5

        # P5 -> P4
        high = td[0]       # P5
        low_orig = feats[1]  # P4
        i = 0  # 索引
        up = self.ups[i]
        if up is not None:
            high_up = up(low_orig, high)      # 输出通道 = c5
            high_up = self.up_proj[i](high_up)  # c5 -> c4
        else:
            high_aligned = self.lateral_convs[i](high)   # c5 -> c4
            high_up = F.interpolate(high_aligned, size=low_orig.shape[-2:], mode='nearest')
        td_feat = self.td_nodes[i]([low_orig, high_up])
        if self.refines_td:
            td_feat = self.refines_td[i](td_feat)
        td[1] = td_feat   # P4_td

        # P4_td -> P3
        high = td[1]       # P4_td
        low_orig = feats[2]  # P3
        i = 1
        up = self.ups[i]
        if up is not None:
            high_up = up(low_orig, high)      # 输出通道 = c4
            high_up = self.up_proj[i](high_up)  # c4 -> c3
        else:
            high_aligned = self.lateral_convs[i](high)   # c4 -> c3
            high_up = F.interpolate(high_aligned, size=low_orig.shape[-2:], mode='nearest')
        td_feat = self.td_nodes[i]([low_orig, high_up])
        if self.refines_td:
            td_feat = self.refines_td[i](td_feat)
        td[2] = td_feat   # P3_td

        # ---------- 自底向上 ----------
        out = [None, None, None]
        out[0] = td[2]   # P3_td

        # P3 -> P4
        low_td = out[0]        # P3_td
        high_td = td[1]        # P4_td
        low_down = self.downs[0](low_td)  # P3_td -> P4 size, 通道自动对齐到 c4
        bu_feat = self.bu_nodes[0]([high_td, low_down])
        if self.refines_bu:
            bu_feat = self.refines_bu[1](bu_feat)  # 注意索引：bu_nodes 的输出对应 out[1]，refines_bu[1] 是正确的
        out[1] = bu_feat        # P4_out

        # P4 -> P5
        low_td = out[1]         # P4_out
        high_td = td[0]         # P5
        low_down = self.downs[1](low_td)  # P4_out -> P5 size, 通道对齐到 c5
        bu_feat = self.bu_nodes[1]([high_td, low_down])
        if self.refines_bu:
            bu_feat = self.refines_bu[2](bu_feat)
        out[2] = bu_feat        # P5_out

        # 返回 P3_out, P4_out, P5_out
        return [out[0], out[1], out[2]]


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 测试三尺度（无 P2）
    print("==== 三尺度 BiFPN (无 P2) ====")
    bifpn = BiFPN([64, 128, 256]).to(device)
    p3 = torch.randn(2, 64, 80, 80).to(device)
    p4 = torch.randn(2, 128, 40, 40).to(device)
    p5 = torch.randn(2, 256, 20, 20).to(device)
    out = bifpn([p3, p4, p5])
    for i, o in enumerate(out):
        print(f"P{i+3}: {o.shape}")   # 应为 (2,64,80,80) (2,128,40,40) (2,256,20,20)

    # 测试四尺度（P2 融合）
    print("\n==== 四尺度 BiFPN (P2 融合) ====")
    bifpn4 = BiFPN([32, 64, 128, 256], fuse_p2=True, p2_to_p3_down="conv").to(device)
    p2 = torch.randn(2, 32, 160, 160).to(device)
    p3 = torch.randn(2, 64, 80, 80).to(device)
    p4 = torch.randn(2, 128, 40, 40).to(device)
    p5 = torch.randn(2, 256, 20, 20).to(device)
    out = bifpn4([p2, p3, p4, p5])
    for i, o in enumerate(out):
        print(f"P{i+3}: {o.shape}")