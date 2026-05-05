import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv


class BiFPN(nn.Module):
    """
    轻量双向特征金字塔 + 固定半层输出 (128, 64, 32)
    输入: [P2, P3, P4, P5]  (尺寸: 160, 80, 40, 20)
    输出: [P2.5, P3.5, P4.5] (尺寸: 128, 64, 32)
    输出通道: P2.5 = c3//2 (128), P3.5 = c4 (256), P4.5 = c5 (512)

    参数:
        channels : [c2, c3, c4, c5] 输入通道数
        mid_ch   : 半层融合时的低维共享通道数，默认 64 (可调小以进一步压缩计算量)
    """
    def __init__(self, channels, mid_ch=64):
        super().__init__()
        c2, c3, c4, c5 = channels
        self.out_c2 = c3 // 2   # 128 (对齐标准检测头 P3 的期望通道)
        self.out_c3 = c4        # 256
        self.out_c4 = c5        # 512
        self.mid_ch = mid_ch

        # ---------- 横向连接 (1x1 卷积) ----------
        self.td_conv5 = nn.Conv2d(c5, c4, 1, bias=False)   # P5 -> P4
        self.td_conv4 = nn.Conv2d(c4, c3, 1, bias=False)   # P4 -> P3
        self.td_conv3 = nn.Conv2d(c3, c2, 1, bias=False)   # P3 -> P2

        # ---------- 深度可分离下采样 ----------
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

        # ---------- 共享低维投影 (降维) ----------
        self.proj_down2 = Conv(c2, mid_ch, 1, act=False)
        self.proj_down3 = Conv(c3, mid_ch, 1, act=False)
        self.proj_down4 = Conv(c4, mid_ch, 1, act=False)
        self.proj_down5 = Conv(c5, mid_ch, 1, act=False)

        # ---------- 输出投影 (升维) ----------
        self.proj_up25 = Conv(mid_ch, self.out_c2, 1, act=False)
        self.proj_up35 = Conv(mid_ch, self.out_c3, 1, act=False)
        self.proj_up45 = Conv(mid_ch, self.out_c4, 1, act=False)

    def forward(self, features):
        p2, p3, p4, p5 = features

        # ========== 自顶向下 (最近邻上采样 + 相加融合) ==========
        p5_up = F.interpolate(self.td_conv5(p5), size=p4.shape[-2:], mode='nearest')
        p4_td = p4 + p5_up

        p4_up = F.interpolate(self.td_conv4(p4_td), size=p3.shape[-2:], mode='nearest')
        p3_td = p3 + p4_up

        p3_up = F.interpolate(self.td_conv3(p3_td), size=p2.shape[-2:], mode='nearest')
        p2_td = p2 + p3_up

        # ========== 自底向上 ==========
        p2_down = self.down2(p2_td)                         # 160 -> 80
        p3_bu = p3_td + p2_down

        p3_down = self.down3(p3_bu)                         # 80 -> 40
        p4_bu = p4_td + p3_down

        p4_down = self.down4(p4_bu)                         # 40 -> 20
        p5_bu = p5 + p4_down

        # ========== 半层生成 (共享低维投影 + 最近邻插值 + 相加) ==========
        f2 = self.proj_down2(p2_td)                         # (B,mid_ch,160)
        f3 = self.proj_down3(p3_bu)                         # (B,mid_ch,80)
        f4 = self.proj_down4(p4_bu)                         # (B,mid_ch,40)
        f5 = self.proj_down5(p5_bu)                         # (B,mid_ch,20)

        # 插值到目标尺寸并相加融合
        p25_low = F.interpolate(f2, size=(128,128), mode='nearest') + \
                  F.interpolate(f3, size=(128,128), mode='nearest')
        p35_low = F.interpolate(f3, size=(64,64), mode='nearest') + \
                  F.interpolate(f4, size=(64,64), mode='nearest')
        p45_low = F.interpolate(f4, size=(32,32), mode='nearest') + \
                  F.interpolate(f5, size=(32,32), mode='nearest')

        # 升维到目标输出通道
        p25 = self.proj_up25(p25_low)   # (B, 128, 128, 128)
        p35 = self.proj_up35(p35_low)   # (B, 256, 64, 64)
        p45 = self.proj_up45(p45_low)   # (B, 512, 32, 32)

        return [p25, p35, p45]


# ====================== 测试 ======================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}\n")

    # YOLO11s 缩放后实际通道 (width=0.5)
    channels = [64, 256, 256, 512]   # P2, P3, P4, P5
    bifpn = BiFPN(channels, mid_ch=64).to(device)

    bs = 2
    p2 = torch.randn(bs, 64, 160, 160).to(device)
    p3 = torch.randn(bs, 256, 80, 80).to(device)
    p4 = torch.randn(bs, 256, 40, 40).to(device)
    p5 = torch.randn(bs, 512, 20, 20).to(device)

    out = bifpn([p2, p3, p4, p5])

    expected = [
        (bs, 128, 128, 128),   # P2.5
        (bs, 256, 64, 64),     # P3.5
        (bs, 512, 32, 32),     # P4.5
    ]

    print("=== BiFPN 输出形状 ===")
    for name, feat, exp in zip(["P2.5", "P3.5", "P4.5"], out, expected):
        print(f"{name}: {feat.shape} (期望 {exp})")
        assert feat.shape == exp, f"形状不匹配！"

    print("\n✅ 所有输出形状验证通过！")
    print(f"总参数量: {sum(p.numel() for p in bifpn.parameters()):,}")