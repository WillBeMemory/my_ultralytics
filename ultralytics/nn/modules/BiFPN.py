import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import HWD
from ultralytics.nn.modules.block import C3k2


class BiFPN(nn.Module):
    """
    子带双向融合 Neck：
    - 自顶向下：对高层子带堆叠进行逆小波上采样，与低层各子带分别融合，
                 保持子带分离。
    - 自底向上：将子带堆叠转换为普通特征图，通过 HWD 下采样 + Concat + C3k2。
    输出三个尺度的普通特征图，可直接连接检测头。
    """
    def __init__(self, channels, c3k2_kwargs=None):
        super().__init__()
        c3, c4, c5 = channels
        self.c3, self.c4, self.c5 = c3, c4, c5
        self.sub3, self.sub4, self.sub5 = c3 // 4, c4 // 4, c5 // 4   # 保存为属性

        # ---- 通道对齐投影 ----
        self.proj_p5 = nn.Conv2d(self.sub5, self.sub4, 1, bias=False)
        self.proj_p4 = nn.Conv2d(self.sub4, self.sub3, 1, bias=False)

        # ---- 各子带融合层 (2*sub_ch -> sub_ch) ----
        self.fuse_ll_p4 = nn.Conv2d(self.sub4 * 2, self.sub4, 1, bias=False)
        self.fuse_lh_p4 = nn.Conv2d(self.sub4 * 2, self.sub4, 1, bias=False)
        self.fuse_hl_p4 = nn.Conv2d(self.sub4 * 2, self.sub4, 1, bias=False)
        self.fuse_hh_p4 = nn.Conv2d(self.sub4 * 2, self.sub4, 1, bias=False)

        self.fuse_ll_p3 = nn.Conv2d(self.sub3 * 2, self.sub3, 1, bias=False)
        self.fuse_lh_p3 = nn.Conv2d(self.sub3 * 2, self.sub3, 1, bias=False)
        self.fuse_hl_p3 = nn.Conv2d(self.sub3 * 2, self.sub3, 1, bias=False)
        self.fuse_hh_p3 = nn.Conv2d(self.sub3 * 2, self.sub3, 1, bias=False)

        # ---- 子带→普通特征转换（用于自底向上） ----
        self.to_feat_p3 = nn.Conv2d(c3, c3, 1, bias=False)
        self.to_feat_p4 = nn.Conv2d(c4, c4, 1, bias=False)
        self.to_feat_p5 = nn.Conv2d(c5, c5, 1, bias=False)

        # ---- 自底向上模块 ----
        kwa = c3k2_kwargs if c3k2_kwargs else dict(n=1, shortcut=False, e=0.5)
        self.down1 = HWD(c3, c4)
        self.down2 = HWD(c4, c5)
        self.p4_out_c3k2 = C3k2(c4 + c4, c4, **kwa)
        self.p5_out_c3k2 = C3k2(c5 + c5, c5, **kwa)

    def _idwt(self, x):
        B, C4, H, W = x.shape
        sub_c = C4 // 4
        x = x.view(B, sub_c, 4, H, W).reshape(B, sub_c * 4, H, W)
        return F.pixel_shuffle(x, 2)

    def forward(self, features):
        p3, p4, p5 = features

        # ================= 自顶向下 =================
        # P5 → P4
        p5_up = self._idwt(p5)                              # (B, self.sub5, 40, 40)
        p5_up = self.proj_p5(p5_up)                         # (B, self.sub4, 40, 40)
        ll4 = self.fuse_ll_p4(torch.cat([p4[:, :self.sub4], p5_up], dim=1))
        lh4 = self.fuse_lh_p4(torch.cat([p4[:, self.sub4:2*self.sub4], p5_up], dim=1))
        hl4 = self.fuse_hl_p4(torch.cat([p4[:, 2*self.sub4:3*self.sub4], p5_up], dim=1))
        hh4 = self.fuse_hh_p4(torch.cat([p4[:, 3*self.sub4:], p5_up], dim=1))
        p4_td = torch.cat([ll4, lh4, hl4, hh4], dim=1)      # (B, self.c4, 40, 40)

        # P4_td → P3
        p4_up = self._idwt(p4_td)                            # (B, self.sub4, 80, 80)
        p4_up = self.proj_p4(p4_up)                          # (B, self.sub3, 80, 80)
        ll3 = self.fuse_ll_p3(torch.cat([p3[:, :self.sub3], p4_up], dim=1))
        lh3 = self.fuse_lh_p3(torch.cat([p3[:, self.sub3:2*self.sub3], p4_up], dim=1))
        hl3 = self.fuse_hl_p3(torch.cat([p3[:, 2*self.sub3:3*self.sub3], p4_up], dim=1))
        hh3 = self.fuse_hh_p3(torch.cat([p3[:, 3*self.sub3:], p4_up], dim=1))
        p3_td = torch.cat([ll3, lh3, hl3, hh3], dim=1)      # (B, self.c3, 80, 80)

        # ================= 自底向上 =================
        p3_feat = self.to_feat_p3(p3_td)
        p4_feat = self.to_feat_p4(p4_td)
        p5_feat = self.to_feat_p5(p5)   # P5 也是子带堆叠，先转为普通特征

        p3_down = self.down1(p3_feat)                        # (B, self.c4, 40, 40)
        p4_out = self.p4_out_c3k2(torch.cat([p4_feat, p3_down], dim=1))

        p4_down = self.down2(p4_out)                         # (B, self.c5, 20, 20)
        p5_out = self.p5_out_c3k2(torch.cat([p5_feat, p4_down], dim=1))

        return [p3_feat, p4_out, p5_out]