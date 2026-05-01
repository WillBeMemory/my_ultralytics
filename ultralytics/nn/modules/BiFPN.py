import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import C3k2


class BiFPN(nn.Module):
    """
    仅 LL 融合 + 与原始 LL 拼接 + 轻量 C3k2 精炼。
    对高层上采样后的 LL 与低层 LL 进行融合，再将融合结果与原始低层 LL 拼接，
    通过 1×1 卷积压缩回原 LL 通道数。保留其他高频子带不变。
    """
    def __init__(self, channels, c3k2_kwargs=None):
        super().__init__()
        c3, c4, c5 = channels
        self.c3, self.c4, self.c5 = c3, c4, c5
        self.sub_c3, self.sub_c4, self.sub_c5 = c3 // 4, c4 // 4, c5 // 4

        # 通道对齐投影（用于高层 LL 对齐到低层 LL 通道数）
        self.proj_sub5 = nn.Conv2d(self.sub_c5, self.sub_c4, 1, bias=False)
        self.proj_sub4 = nn.Conv2d(self.sub_c4, self.sub_c3, 1, bias=False)

        # LL 融合层：输入 2*sub_c4，输出 sub_c4
        self.fuse_ll_p4 = nn.Conv2d(self.sub_c4 * 2, self.sub_c4, 1, bias=False)
        self.fuse_ll_p3 = nn.Conv2d(self.sub_c3 * 2, self.sub_c3, 1, bias=False)

        # LL 压缩层：将拼接后的双倍 LL 通道压缩回单倍
        self.compress_ll_p4 = nn.Conv2d(self.sub_c4 * 2, self.sub_c4, 1, bias=False)
        self.compress_ll_p3 = nn.Conv2d(self.sub_c3 * 2, self.sub_c3, 1, bias=False)

        # 轻量精炼 C3k2（输入输出通道均为 c4 / c3）
        kwa = c3k2_kwargs if c3k2_kwargs else dict(n=1, shortcut=False, e=0.25)
        self.refine_p4 = C3k2(c4, c4, **kwa)
        self.refine_p3 = C3k2(c3, c3, **kwa)

    def _idwt(self, x):
        """逆 Haar 小波上采样 (pixel shuffle)，x: (B, c, H, W), c=4*sub_c"""
        B, C4, H, W = x.shape
        sub_c = C4 // 4
        x = x.view(B, sub_c, 4, H, W).reshape(B, sub_c * 4, H, W)
        return F.pixel_shuffle(x, 2)            # (B, sub_c, 2H, 2W)

    def forward(self, features):
        p3, p4, p5 = features                   # 均为子带堆叠格式

        # ----- P5 → P4 -----
        p5_up = self._idwt(p5)                  # (B, sub_c5, H4, W4)
        p5_up = self.proj_sub5(p5_up)           # (B, sub_c4, H4, W4) 通道对齐

        # LL 融合 + 与原始 LL 拼接
        ll_fused = self.fuse_ll_p4(torch.cat([p4[:, :self.sub_c4], p5_up], dim=1))  # (B, sub_c4, H4, W4)
        ll_cat = torch.cat([ll_fused, p4[:, :self.sub_c4]], dim=1)                  # (B, 2*sub_c4, H4, W4)
        ll_out = self.compress_ll_p4(ll_cat)     # (B, sub_c4, H4, W4)

        # 与其他高频子带重组，保持子带顺序
        p4_fused = torch.cat([ll_out, p4[:, self.sub_c4:]], dim=1)   # (B, c4, H4, W4)
        p4_out = self.refine_p4(p4_fused)        # 轻量精炼

        # ----- P4 → P3 -----
        p4_up = self._idwt(p4_out)               # (B, sub_c4, H3, W3)
        p4_up = self.proj_sub4(p4_up)            # (B, sub_c3, H3, W3)

        ll_fused = self.fuse_ll_p3(torch.cat([p3[:, :self.sub_c3], p4_up], dim=1))
        ll_cat = torch.cat([ll_fused, p3[:, :self.sub_c3]], dim=1)
        ll_out = self.compress_ll_p3(ll_cat)
        p3_fused = torch.cat([ll_out, p3[:, self.sub_c3:]], dim=1)
        p3_out = self.refine_p3(p3_fused)

        return [p3_out, p4_out, p5]


# 测试
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    c3, c4, c5 = 64, 128, 256
    model = BiFPN([c3, c4, c5]).to(device)

    p3 = torch.randn(2, c3, 80, 80).to(device)
    p4 = torch.randn(2, c4, 40, 40).to(device)
    p5 = torch.randn(2, c5, 20, 20).to(device)

    out = model([p3, p4, p5])
    for name, feat in zip(["P3", "P4", "P5"], out):
        print(f"{name}: {feat.shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")