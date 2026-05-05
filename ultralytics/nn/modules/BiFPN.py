import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv


class BiFPN(nn.Module):
    """
    极速双向特征金字塔 + 半层生成。
    输入： [P2, P3, P4, P5]  (尺寸：160, 80, 40, 20)
    输出： [P2.5, P3.5, P4.5] (尺寸：120, 60, 30)
    输出通道：P2.5 = c3//2, P3.5 = c4, P4.5 = c5
    """
    def __init__(self, channels, out_channels=None,
                 use_c3k2=False, c3k2_kwargs=None,
                 mid_ch=64):
        super().__init__()
        assert len(channels) == 4, "需提供4个尺度 [c2,c3,c4,c5]"
        c2, c3, c4, c5 = channels
        self.out_c2 = c3 // 2   # 128
        self.out_c3 = c4        # 256
        self.out_c4 = c5        # 512
        self.mid_ch = mid_ch

        # ---------- 横向连接 (1x1 卷积) ----------
        self.td_conv5 = nn.Conv2d(c5, c4, 1, bias=False)
        self.td_conv4 = nn.Conv2d(c4, c3, 1, bias=False)
        self.td_conv3 = nn.Conv2d(c3, c2, 1, bias=False)

        # ---------- 下采样（深度可分离，保持轻量） ----------
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c2, 3, stride=2, padding=1, groups=c2, bias=False),
            nn.Conv2d(c2, c3, 1, bias=False),
            nn.BatchNorm2d(c3),
            nn.SiLU(inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(c3, c3, 3, stride=2, padding=1, groups=c3, bias=False),
            nn.Conv2d(c3, c4, 1, bias=False),
            nn.BatchNorm2d(c4),
            nn.SiLU(inplace=True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(c4, c4, 3, stride=2, padding=1, groups=c4, bias=False),
            nn.Conv2d(c4, c5, 1, bias=False),
            nn.BatchNorm2d(c5),
            nn.SiLU(inplace=True)
        )

        # ---------- 共享低维投影 ----------
        self.proj_down2 = Conv(c2, mid_ch, 1, act=False)
        self.proj_down3 = Conv(c3, mid_ch, 1, act=False)
        self.proj_down4 = Conv(c4, mid_ch, 1, act=False)
        self.proj_down5 = Conv(c5, mid_ch, 1, act=False)

        # ---------- 输出投影 ----------
        self.proj_up25 = Conv(mid_ch, self.out_c2, 1, act=False)
        self.proj_up35 = Conv(mid_ch, self.out_c3, 1, act=False)
        self.proj_up45 = Conv(mid_ch, self.out_c4, 1, act=False)

        # ---------- 可选后处理（默认关闭） ----------
        kwa = c3k2_kwargs if c3k2_kwargs else dict(n=1, shortcut=False, e=0.5)
        if use_c3k2:
            from ultralytics.nn.modules.block import C3k2
            self.refine_p4_td = C3k2(c4, c4, **kwa)
            self.refine_p3_td = C3k2(c3, c3, **kwa)
            self.refine_p2_td = C3k2(c2, c2, **kwa)
            self.refine_p3_bu = C3k2(c3, c3, **kwa)
            self.refine_p4_bu = C3k2(c4, c4, **kwa)
            self.refine_p5_bu = C3k2(c5, c5, **kwa)
        else:
            self.refine_p4_td = self.refine_p3_td = self.refine_p2_td = nn.Identity()
            self.refine_p3_bu = self.refine_p4_bu = self.refine_p5_bu = nn.Identity()

    def forward(self, features):
        p2, p3, p4, p5 = features

        # ========== 自顶向下（使用最近邻插值 + 直接相加） ==========
        p5_up = F.interpolate(self.td_conv5(p5), size=p4.shape[-2:], mode='nearest')
        p4_td = self.refine_p4_td(p4 + p5_up)

        p4_up = F.interpolate(self.td_conv4(p4_td), size=p3.shape[-2:], mode='nearest')
        p3_td = self.refine_p3_td(p3 + p4_up)

        p3_up = F.interpolate(self.td_conv3(p3_td), size=p2.shape[-2:], mode='nearest')
        p2_td = self.refine_p2_td(p2 + p3_up)

        # ========== 自底向上 ==========
        p2_down = self.down2(p2_td)
        p3_bu = self.refine_p3_bu(p3_td + p2_down)

        p3_bu_down = self.down3(p3_bu)
        p4_bu = self.refine_p4_bu(p4_td + p3_bu_down)

        p4_bu_down = self.down4(p4_bu)
        p5_bu = self.refine_p5_bu(p5 + p4_bu_down)

        # ========== 半层生成（最近邻插值 + 直接相加） ==========
        f2 = self.proj_down2(p2_td)
        f3 = self.proj_down3(p3_bu)
        f4 = self.proj_down4(p4_bu)
        f5 = self.proj_down5(p5_bu)

        p25_low = F.interpolate(f2, size=(120, 120), mode='nearest') + \
                  F.interpolate(f3, size=(120, 120), mode='nearest')
        p35_low = F.interpolate(f3, size=(60, 60), mode='nearest') + \
                  F.interpolate(f4, size=(60, 60), mode='nearest')
        p45_low = F.interpolate(f4, size=(30, 30), mode='nearest') + \
                  F.interpolate(f5, size=(30, 30), mode='nearest')

        p25 = self.proj_up25(p25_low)
        p35 = self.proj_up35(p35_low)
        p45 = self.proj_up45(p45_low)

        return [p25, p35, p45]


# ====================== 测试 ======================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}\n")

    channels = [64, 256, 256, 512]   # 缩放后实际通道
    bifpn = BiFPN(channels, mid_ch=64, use_c3k2=False).to(device)

    bs = 2
    p2 = torch.randn(bs, 64, 160, 160).to(device)
    p3 = torch.randn(bs, 256, 80, 80).to(device)
    p4 = torch.randn(bs, 256, 40, 40).to(device)
    p5 = torch.randn(bs, 512, 20, 20).to(device)

    out = bifpn([p2, p3, p4, p5])

    expected = [
        (bs, 128, 120, 120),   # P2.5
        (bs, 256, 60, 60),     # P3.5
        (bs, 512, 30, 30),     # P4.5
    ]

    print("=== BiFPN 输出形状 ===")
    for name, feat, exp in zip(["P2.5", "P3.5", "P4.5"], out, expected):
        print(f"{name}: {feat.shape} (期望 {exp})")
        assert feat.shape == exp, f"形状不匹配！"

    print("\n✅ 所有输出形状验证通过！")
    print(f"总参数量: {sum(p.numel() for p in bifpn.parameters()):,}")