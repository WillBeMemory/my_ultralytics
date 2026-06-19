import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv, C3k2


class BiFPN_Add(nn.Module):
    """快速归一化加权融合（BiFPN 核心，ReLU 归一化）"""
    def __init__(self, num_inputs=2):
        super().__init__()
        self.w = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.eps = 1e-4

    def forward(self, xs):
        w = F.relu(self.w)
        w_norm = w / (w.sum() + self.eps)
        return sum(w_norm[i] * xs[i] for i in range(len(xs)))


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积 — 仅 DW 3×3，通道混合由前置 mid_conv(1×1) 完成"""
    def __init__(self, ch, kernel_size=3, act=True):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, kernel_size, stride=1,
                            padding=kernel_size // 2, groups=ch, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.dw(x)))


class FPN_PAN(nn.Module):
    """
    标准 FPN-PAN，与 YOLO11 默认 Neck 结构完全一致：
    - FPN (Top-Down):   Upsample → Concat → C3k2
    - PAN (Bottom-Up):  Conv(s=2) → Concat → C3k2
    - C3k2 内部 n=1（YOLO11 默认，n 控制 Bottleneck 数量）
    """
    def __init__(self, channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = channels
        self._out_channels = out_channels
        self._initialized = False

    def _init_layers(self, p2, p3, p4):
        o2, o3, o4 = self._out_channels

        # ========== FPN (Top-Down) ==========
        # P4 → Upsample → Concat(P3) → C3k2 → P3_fpn
        self.fpn_p3 = C3k2(p3.shape[1] + p4.shape[1], o3, n=1, c3k=False)
        # P3_fpn → Upsample → Concat(P2) → C3k2 → P2_fpn
        self.fpn_p2 = C3k2(p2.shape[1] + o3, o2, n=1, c3k=False)

        # ========== PAN (Bottom-Up) ==========
        # P2_fpn → Conv(s=2) → Concat(P3_fpn) → C3k2 → P3_pan
        self.pan_p2_down = Conv(o2, o3, 3, 2)
        self.pan_p3 = C3k2(o3 + o3, o3, n=1, c3k=False)
        # P3_pan → Conv(s=2) → Concat(P4) → C3k2 → P4_pan
        self.pan_p3_down = Conv(o3, o4, 3, 2)
        self.pan_p4 = C3k2(o4 + p4.shape[1], o4, n=1, c3k=False)

        self.to(p2.device)
        self._initialized = True

    def forward(self, features):
        p2, p3, p4 = features

        if not self._initialized:
            self._init_layers(p2, p3, p4)

        # ========== FPN (Top-Down) ==========
        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode='nearest')
        p3_fpn = self.fpn_p3(torch.cat([p3, p4_up], dim=1))

        p3_up = F.interpolate(p3_fpn, size=p2.shape[-2:], mode='nearest')
        p2_fpn = self.fpn_p2(torch.cat([p2, p3_up], dim=1))

        # ========== PAN (Bottom-Up) ==========
        p2_down = self.pan_p2_down(p2_fpn)
        p3_pan = self.pan_p3(torch.cat([p3_fpn, p2_down], dim=1))

        p3_down = self.pan_p3_down(p3_pan)
        p4_pan = self.pan_p4(torch.cat([p4, p3_down], dim=1))

        return [p2_fpn, p3_pan, p4_pan]


class BiFPNLayer(nn.Module):
    """
    BiFPN 单层，P2/P3/P4 两两加权融合（Top-Down + Bottom-Up）
    - 使用 BiFPN_Add 做加权融合（而非 Concat）
    - 1x1 Conv 做通道投影保证维度一致
    - DWConv 做融合后精炼
    """
    def __init__(self, channels=None, use_refine=True):
        super().__init__()
        self._channels = channels
        self.use_refine = use_refine
        self._initialized = False

    def _init_layers(self, p2, p3, p4):
        c2, c3, c4 = p2.shape[1], p3.shape[1], p4.shape[1]

        # ========== Top-Down：P4→P3, P3→P2 ==========
        # P4 → 1x1 Conv(512→256) → Upsample → BiFPN_Add(P3, P4_up) → DWConv refine
        self.td_p4_to_p3 = Conv(c4, c3, 1, act=False)
        self.td_p3_fuse = BiFPN_Add(2)
        self.td_p3_refine = DepthwiseSeparableConv(c3) if self.use_refine else nn.Identity()

        # P3_td → 1x1 Conv(256→128) → Upsample → BiFPN_Add(P2, P3_up) → DWConv refine
        self.td_p3_to_p2 = Conv(c3, c2, 1, act=False)
        self.td_p2_fuse = BiFPN_Add(2)
        self.td_p2_refine = DepthwiseSeparableConv(c2) if self.use_refine else nn.Identity()

        # ========== Bottom-Up：P2→P3, P3→P4 ==========
        # P2_td → Conv(s=2, 128→256) → BiFPN_Add(P3_td, P2_down) → DWConv refine
        self.bu_p2_to_p3 = Conv(c2, c3, 3, 2)
        self.bu_p3_fuse = BiFPN_Add(2)
        self.bu_p3_refine = DepthwiseSeparableConv(c3) if self.use_refine else nn.Identity()

        # P3_out → Conv(s=2, 256→512) → BiFPN_Add(P4, P3_down) → DWConv refine
        self.bu_p3_to_p4 = Conv(c3, c4, 3, 2)
        self.bu_p4_fuse = BiFPN_Add(2)
        self.bu_p4_refine = DepthwiseSeparableConv(c4) if self.use_refine else nn.Identity()

        self.to(p2.device)
        self._initialized = True

    def forward(self, p2_in, p3_in, p4_in):
        if not self._initialized:
            self._init_layers(p2_in, p3_in, p4_in)

        # ========== Top-Down ==========
        # P4 → P3 融合
        p4_up = F.interpolate(self.td_p4_to_p3(p4_in), size=p3_in.shape[-2:], mode='nearest')
        p3_td = self.td_p3_fuse([p3_in, p4_up])
        p3_td = self.td_p3_refine(p3_td)

        # P3 → P2 融合
        p3_up = F.interpolate(self.td_p3_to_p2(p3_td), size=p2_in.shape[-2:], mode='nearest')
        p2_td = self.td_p2_fuse([p2_in, p3_up])
        p2_td = self.td_p2_refine(p2_td)

        # ========== Bottom-Up ==========
        # P2 → P3 融合
        p2_down = self.bu_p2_to_p3(p2_td)
        p3_out = self.bu_p3_fuse([p3_td, p2_down])
        p3_out = self.bu_p3_refine(p3_out)

        # P3 → P4 融合
        p3_down = self.bu_p3_to_p4(p3_out)
        p4_out = self.bu_p4_fuse([p4_in, p3_down])
        p4_out = self.bu_p4_refine(p4_out)

        return p2_td, p3_out, p4_out


class AlignFuseBroadcast(nn.Module):
    """
    对齐-共识-广播（Align-Fuse-Broadcast）融合层。
    接 FPN-PAN 输出 [P2, P3, P4] 之后，以 P3 为枢纽做一次跨尺度“共识 → 广播”：

    Stage 1（共识）：P2 下采样、P4 上采样，均对齐到 P3 尺度/通道，与 P3 三路加权融合
        （BiFPN_Add，初始 ones→均匀平均）→ 共识特征 F_mid（P3 尺度）。
    Stage 2（广播，broadcast=True）：F_mid 上采样与原始 P2 加权融合→out_P2；
        F_mid 下采样与原始 P4 加权融合→out_P4；out_P3 = F_mid。
        broadcast=False 时退化：out_P2=P2_orig、out_P3=F_mid、out_P4=P4_orig（仅共识）。

    设计约束（与既有消融结论对齐）：
      - 全程加权融合（BiFPN_Add），初始=均匀平均；不上 C3k2（避免过平滑小目标边缘）；
      - 可选残差式 1×1 精炼共识 F_mid（refine 开关，零初始化→identity，保边不空间平滑）；
      - Stage2 保留原始 P2/P4 作残差一路（保定位精度，out_P2/out_P4 都有原始特征）；
      - out_P3 直接用 F_mid，不再二次融合（避免冗余，对应“加 3rd-path hurt”教训）。
    新建随机初始化 Conv 时做 RNG 隔离（见记忆 rng-isolation-on-module-edit）。

    注：broadcast=True 初始会把共识再混回 P2/P4，对 P2 分布的初始搅动比纯 BiFPNLayer 大；
    BiFPN_Add 是全局均匀（非随机剧烈扰动），不会触发“随机权重→≈0.5x”的失真，
    但仍建议跑满 >100ep 再判方向（见记忆 dont-judge-modules-early）。
    """
    def __init__(self, channels=None, broadcast=True, refine=False):
        super().__init__()
        self._channels = channels
        self.broadcast = broadcast
        self.refine = refine
        self._initialized = False

    def _init_layers(self, p2, p3, p4):
        c2, c3, c4 = p2.shape[1], p3.shape[1], p4.shape[1]

        # RNG 隔离：本层新建若干随机初始化 Conv，save/restore 全局 RNG，保持对后续初始化中性
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        try:
            # ---------- Stage 1：对齐到 P3 尺度，加权融合出共识 F_mid ----------
            # P2 → 下采样（stride=2 → P3 尺度）+ 通道 c2→c3
            self.s1_p2_down = Conv(c2, c3, 3, 2)
            # P4 → 通道 c4→c3（2× 上采样在 forward 用 interpolate 完成）
            self.s1_p4_up_proj = Conv(c4, c3, 1, act=False)
            # 三路加权融合：[P3, P2↓, P4↑]，BiFPN_Add 初始 ones→均匀平均
            self.s1_fuse = BiFPN_Add(3)

            if self.refine:
                # ---------- Stage 1.5：对共识 F_mid 残差精炼 ----------
                # f_mid_refined = f_mid + SiLU(Conv1x1(f_mid))；Conv 零初始化 → 初始 Δ=0 = identity。
                # 1x1 只做通道重组、不空间平滑 → 保小目标边缘（避开 use_refine/C3k2 过平滑前科）；
                # 残差 + 零初始化 → 训练前期对 F_mid 及其广播输出零扰动（见 attention-module-integration-design）。
                self.s15_refine = nn.Sequential(
                    nn.Conv2d(c3, c3, 1, bias=False),
                    nn.SiLU(),
                )
                nn.init.zeros_(self.s15_refine[0].weight)   # 初始 Δ=0 → f_mid_refined == f_mid

            if self.broadcast:
                # ---------- Stage 2：把共识 F_mid 广播回 P2/P4，与原始特征加权融合 ----------
                # F_mid → 通道 c3→c2（2× 上采样在 forward 用 interpolate 完成）
                self.s2_mid_to_p2_proj = Conv(c3, c2, 1, act=False)
                self.s2_fuse_p2 = BiFPN_Add(2)    # [P2_orig, up(F_mid→P2)]
                # F_mid → 下采样（stride=2 → P4 尺度）+ 通道 c3→c4
                self.s2_mid_to_p4_down = Conv(c3, c4, 3, 2)
                self.s2_fuse_p4 = BiFPN_Add(2)    # [P4_orig, down(F_mid→P4)]
            # out_P3 = F_mid（直接用，不再二次融合）

            self.to(p2.device)
            self._initialized = True
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state)

    def forward(self, p2, p3, p4):
        if not self._initialized:
            self._init_layers(p2, p3, p4)

        # ---------- Stage 1：共识 ----------
        p2_down = self.s1_p2_down(p2)                                                # → P3 尺度, c3
        p4_up = F.interpolate(self.s1_p4_up_proj(p4), size=p3.shape[-2:], mode='nearest')  # → P3 尺度, c3
        f_mid = self.s1_fuse([p3, p2_down, p4_up])                                    # P3 尺度, c3

        if self.refine:
            # Stage 1.5：残差精炼共识（零初始化 → 初始 = identity，不扰动）
            f_mid = f_mid + self.s15_refine(f_mid)

        if not self.broadcast:
            # 退化：仅输出共识，P2/P4 保持原始
            return p2, f_mid, p4

        # ---------- Stage 2：广播 ----------
        # 共识 → P2 尺度，与原始 P2 加权融合
        mid_to_p2 = F.interpolate(self.s2_mid_to_p2_proj(f_mid), size=p2.shape[-2:], mode='nearest')  # → P2 尺度, c2
        out_p2 = self.s2_fuse_p2([p2, mid_to_p2])

        # 共识 → P4 尺度，与原始 P4 加权融合
        mid_to_p4 = self.s2_mid_to_p4_down(f_mid)                                    # → P4 尺度, c4
        out_p4 = self.s2_fuse_p4([p4, mid_to_p4])

        # P3 直接用共识
        return out_p2, f_mid, out_p4


class FPN_PAN_BiFPN(nn.Module):
    """
    FPN-PAN + 后接融合层。
    - FPN-PAN：与 YOLO11 默认结构完全一致（Conv + C3k2 + Upsample + Concat）。
    - 后接融合层由 fuse_mode 选择：
        * 'bifpn'（默认）：BiFPNLayer 顺序 top-down/bottom-up 加权融合，可堆叠多层
          （num_bifpn_layers / use_refine 生效）。与历史行为完全一致，向后兼容。
        * 'afb'：AlignFuseBroadcast，对齐-共识-广播（broadcast 开关生效；
          num_bifpn_layers / use_refine 被忽略）。
    yaml args 末尾追加 fuse_mode(字符串)、broadcast(布尔)，经 tasks.py 的 extra_args=args[2:]
    透传，不受 width/depth 通道缩放影响。
    """
    def __init__(self, channels, out_channels=None, num_bifpn_layers=1, use_refine=False,
                 fuse_mode='bifpn', broadcast=True, refine=False):
        super().__init__()
        c2, c3, c4 = channels
        if out_channels is None:
            out_channels = [c2, c3, c4]
        o2, o3, o4 = out_channels

        self.num_bifpn_layers = num_bifpn_layers
        self.use_refine = use_refine
        self.fuse_mode = fuse_mode
        self.broadcast = broadcast
        self.refine = refine

        # FPN-PAN（标准 YOLO11 结构）
        self.fpn_pan = FPN_PAN(channels, out_channels)

        # 后接融合层
        if fuse_mode == 'afb':
            self.fuse_layer = AlignFuseBroadcast(out_channels, broadcast=broadcast, refine=refine)
            self.bifpn_layers = None
        else:
            self.fuse_layer = None
            if num_bifpn_layers > 0:
                self.bifpn_layers = nn.ModuleList([
                    BiFPNLayer(out_channels, use_refine) for _ in range(num_bifpn_layers)
                ])
            else:
                self.bifpn_layers = None

    def forward(self, features):
        p2, p3, p4 = features

        # FPN-PAN
        p2_out, p3_out, p4_out = self.fpn_pan([p2, p3, p4])

        # 后接融合
        if self.fuse_mode == 'afb':
            p2_out, p3_out, p4_out = self.fuse_layer(p2_out, p3_out, p4_out)
        elif self.bifpn_layers:
            for layer in self.bifpn_layers:
                p2_out, p3_out, p4_out = layer(p2_out, p3_out, p4_out)

        return [p2_out, p3_out, p4_out]


# ================== 测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    bs = 2
    p2 = torch.randn(bs, 128, 160, 160).to(device)
    p3 = torch.randn(bs, 256, 80, 80).to(device)
    p4 = torch.randn(bs, 512, 40, 40).to(device)

    channels = [128, 256, 512]
    out_channels = [128, 256, 512]

    print("\n=== Testing FPN_PAN (标准 YOLO11 结构) ===")
    fpn_pan = FPN_PAN(channels, out_channels).to(device)
    outs = fpn_pan([p2, p3, p4])
    for i, (out, exp) in enumerate(zip(outs, [
        (bs, 128, 160, 160),
        (bs, 256, 80, 80),
        (bs, 512, 40, 40)
    ]), start=2):
        status = "OK" if out.shape == exp else "FAIL"
        print(f"  P{i}: {list(out.shape)} expected {list(exp)} [{status}]")
    total_params = sum(p.numel() for p in fpn_pan.parameters())
    print(f"  FPN_PAN Params: {total_params:,}")

    print("\n=== Testing BiFPNLayer (两两加权融合) ===")
    bifpn_layer = BiFPNLayer(out_channels, use_refine=True).to(device)
    p2_out, p3_out, p4_out = bifpn_layer(p2, p3, p4)
    for name, out, exp in [
        ("P2", p2_out, (bs, 128, 160, 160)),
        ("P3", p3_out, (bs, 256, 80, 80)),
        ("P4", p4_out, (bs, 512, 40, 40))
    ]:
        status = "OK" if out.shape == exp else "FAIL"
        print(f"  {name}: {list(out.shape)} expected {list(exp)} [{status}]")
    print(f"  BiFPNLayer Params: {sum(p.numel() for p in bifpn_layer.parameters()):,}")

    print("\n=== Testing FPN_PAN_BiFPN (1 BiFPN layer) ===")
    fpn_pan_bifpn = FPN_PAN_BiFPN(
        channels, out_channels,
        num_bifpn_layers=1,
        use_refine=True
    ).to(device)
    outs = fpn_pan_bifpn([p2, p3, p4])
    for i, (out, exp) in enumerate(zip(outs, [
        (bs, 128, 160, 160),
        (bs, 256, 80, 80),
        (bs, 512, 40, 40)
    ]), start=2):
        status = "OK" if out.shape == exp else "FAIL"
        print(f"  P{i}: {list(out.shape)} expected {list(exp)} [{status}]")
    total_params = sum(p.numel() for p in fpn_pan_bifpn.parameters())
    print(f"  FPN_PAN_BiFPN Params: {total_params:,}")

    print("\n=== Testing FPN_PAN_BiFPN (3 BiFPN layers) ===")
    fpn_pan_bifpn3 = FPN_PAN_BiFPN(
        channels, out_channels,
        num_bifpn_layers=3,
        use_refine=True
    ).to(device)
    outs = fpn_pan_bifpn3([p2, p3, p4])
    for i, (out, exp) in enumerate(zip(outs, [
        (bs, 128, 160, 160),
        (bs, 256, 80, 80),
        (bs, 512, 40, 40)
    ]), start=2):
        status = "OK" if out.shape == exp else "FAIL"
        print(f"  P{i}: {list(out.shape)} expected {list(exp)} [{status}]")
    total_params = sum(p.numel() for p in fpn_pan_bifpn3.parameters())
    print(f"  FPN_PAN_BiFPN (3 layers) Params: {total_params:,}")