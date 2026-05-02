import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv, Bottleneck


# ---------- 辅助模块 ----------
class ECA(nn.Module):
    """高效通道注意力"""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(dim=(2, 3), keepdim=True)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class StarBlock(nn.Module):
    """轻量星操作块"""
    def __init__(self, c1, c2, k=3, e=0.5, shortcut=False, act=nn.ReLU6(inplace=True)):
        super().__init__()
        hidden = max(1, int(c1 * e))
        self.add = shortcut and c1 == c2
        self.dw1 = nn.Conv2d(c1, c1, k, padding=k // 2, groups=c1, bias=False)
        self.pw1 = nn.Conv2d(c1, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act1 = act
        self.dw2 = nn.Conv2d(c1, c1, k, padding=k // 2, groups=c1, bias=False)
        self.pw2 = nn.Conv2d(c1, hidden, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden, c2, 1, bias=False),
            nn.BatchNorm2d(c2)
        )
        self.out_act = act

    def forward(self, x):
        x1 = self.dw1(x); x1 = self.pw1(x1); x1 = self.bn1(x1); x1 = self.act1(x1)
        x2 = self.dw2(x); x2 = self.pw2(x2); x2 = self.bn2(x2)
        out = x1 * x2
        out = self.fusion(out)
        out = self.out_act(out)
        return out + x if self.add else out


class WaveletTextureSuppress(nn.Module):
    """小波纹理抑制"""
    def __init__(self, c):
        super().__init__()
        self.star = StarBlock(c * 2, 1, k=3, e=0.5, shortcut=False)
        self.bias = nn.Parameter(torch.tensor(2.0))
        self.sigmoid = nn.Sigmoid()

    def forward(self, lh, hl):
        combined = torch.cat([lh, hl], dim=1)
        logit = self.star(combined) + self.bias.view(1, -1, 1, 1)
        mask = self.sigmoid(logit)
        return lh * mask, hl * mask


# ====================== WaveletRefine 完整版 ======================
class WaveletRefine(nn.Module):
    """
    小波域特征精炼 + 下采样模块，支持：
    - 标准模式：全分解 + 融合
    - keep_subbands 模式：子带堆叠输出
    - decompose_ll_only 模式：仅分解 LL1，复用历史高频
    """
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, k=3,
        texture_suppress=True,
        keep_subbands=False,
        use_eca=True,
        high_scale=True,
        cross_subband=False,
        decompose_ll_only=False,
    ):
        super().__init__()
        assert c2 % 4 == 0, "c2 must be divisible by 4"
        self.c1, self.c2 = c1, c2
        self.texture_suppress = texture_suppress
        self.high_scale = high_scale if not texture_suppress else False
        self.keep_subbands = keep_subbands
        self.cross_subband = cross_subband
        self.decompose_ll_only = decompose_ll_only

        # ---- 通用 Haar 分解核 ----
        w_ll = torch.tensor([[1., 1.], [1., 1.]]) / 2.0
        w_lh = torch.tensor([[1., -1.], [1., -1.]]) / 2.0
        w_hl = torch.tensor([[1., 1.], [-1., -1.]]) / 2.0
        w_hh = torch.tensor([[1., -1.], [-1., 1.]]) / 2.0
        haar_init = torch.stack([w_ll, w_lh, w_hl, w_hh], dim=0).unsqueeze(1)  # (4,1,2,2)

        if decompose_ll_only:
            # 本模式输入是四子带堆叠，每个子带占 c1//4 通道
            sub_c = c1 // 4
            self.register_buffer('haar_ll_filter', haar_init.repeat(sub_c, 1, 1, 1))
            # LL 精炼（仅对子带通道 operate）
            self.ll_refine_sub = nn.Sequential(*[
                Bottleneck(sub_c, sub_c, shortcut, g, k=(k, k), e=1.0) for _ in range(n)
            ])
            self.ll_eca_sub = ECA(sub_c) if use_eca else nn.Identity()
            # 高频处理（适用于子带通道）
            if texture_suppress:
                self.texture_module_sub = WaveletTextureSuppress(sub_c)
            elif high_scale:
                self.lh_scale_sub = Conv(sub_c, sub_c, k=1, act=False)
                self.hl_scale_sub = Conv(sub_c, sub_c, k=1, act=False)
                self.hh_scale_sub = Conv(sub_c, sub_c, k=1, act=False)
        else:
            # 常规全分解模式，c1 为全通道数
            self.register_buffer('haar_filter', haar_init.repeat(c1, 1, 1, 1))
            self.ll_refine = nn.Sequential(*[
                Bottleneck(c1, c1, shortcut, g, k=(k, k), e=1.0) for _ in range(n)
            ])
            self.ll_eca = ECA(c1) if use_eca else nn.Identity()
            if texture_suppress:
                self.texture_module = WaveletTextureSuppress(c1)
            elif high_scale:
                self.lh_scale = Conv(c1, c1, k=1, act=False)
                self.hl_scale = Conv(c1, c1, k=1, act=False)
                self.hh_scale = Conv(c1, c1, k=1, act=False)

        # 跨子带交互（仅在常规模式有效）
        if cross_subband and not decompose_ll_only:
            self.high_gate = nn.Sequential(
                nn.Conv2d(c1 * 3, c1, 1, bias=False),
                nn.Sigmoid()
            )

        # ---- 输出投影 ----
        out_sub = c2 // 4
        if decompose_ll_only:
            sub_c = c1 // 4
            self.downsample_high = Conv(sub_c, sub_c, k=3, s=2, g=sub_c, act=False)
            self.ll_proj = Conv(sub_c, out_sub, k=1, act=False)
            self.lh_combiner = Conv(sub_c * 2, out_sub, k=1, act=False)
            self.hl_combiner = Conv(sub_c * 2, out_sub, k=1, act=False)
            self.hh_combiner = Conv(sub_c * 2, out_sub, k=1, act=False)
        elif keep_subbands:
            self.ll_proj = Conv(c1, out_sub, k=1, act=False)
            self.lh_proj = Conv(c1, out_sub, k=1, act=False)
            self.hl_proj = Conv(c1, out_sub, k=1, act=False)
            self.hh_proj = Conv(c1, out_sub, k=1, act=False)
        else:
            self.fuse = Conv(c1 * 4, c2, k=1, act=True)

    def forward(self, x):
        B, C, H, W = x.shape

        # ========== decompose_ll_only 模式 ==========
        if self.decompose_ll_only:
            sub_c = self.c1 // 4  # 每个子带通道数
            ll1 = x[:, :sub_c]
            lh1 = x[:, sub_c:2*sub_c]
            hl1 = x[:, 2*sub_c:3*sub_c]
            hh1 = x[:, 3*sub_c:]

            # 1. 仅对 LL1 分解
            coeff = F.conv2d(ll1, self.haar_ll_filter, stride=2, groups=sub_c)
            coeff = coeff.view(B, sub_c, 4, H//2, W//2)
            ll2 = coeff[:, :, 0]; lh2 = coeff[:, :, 1]; hl2 = coeff[:, :, 2]; hh2 = coeff[:, :, 3]

            # 2. LL 精炼（子带大小）
            ll2 = self.ll_refine_sub(ll2)
            ll2 = self.ll_eca_sub(ll2)

            # 3. 高频处理
            if self.texture_suppress:
                lh2, hl2 = getattr(self, 'texture_module_sub', None)(lh2, hl2)
            elif self.high_scale:
                lh2 = self.lh_scale_sub(lh2); hl2 = self.hl_scale_sub(hl2)
                hh2 = self.hh_scale_sub(hh2) if hasattr(self, 'hh_scale_sub') else hh2

            # 4. 历史高频下采样
            lh1_down = self.downsample_high(lh1)
            hl1_down = self.downsample_high(hl1)
            hh1_down = self.downsample_high(hh1)

            # 5. 合并与投影
            ll_out = self.ll_proj(ll2)
            lh_out = self.lh_combiner(torch.cat([lh2, lh1_down], dim=1))
            hl_out = self.hl_combiner(torch.cat([hl2, hl1_down], dim=1))
            hh_out = self.hh_combiner(torch.cat([hh2, hh1_down], dim=1))

            # 6. 堆叠输出
            out = torch.stack([ll_out, lh_out, hl_out, hh_out], dim=2).reshape(B, self.c2, H//2, W//2)
            return out

        # ========== 常规全分解模式 ==========
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        coeff = F.conv2d(x, self.haar_filter, stride=2, groups=C)
        coeff = coeff.view(B, C, 4, H//2, W//2)
        ll, lh, hl, hh = coeff[:, :, 0], coeff[:, :, 1], coeff[:, :, 2], coeff[:, :, 3]

        # LL 精炼
        ll = self.ll_refine(ll)
        ll = self.ll_eca(ll)

        # 高频处理
        if self.texture_suppress:
            lh, hl = self.texture_module(lh, hl)
        elif self.high_scale:
            lh = self.lh_scale(lh); hl = self.hl_scale(hl)
            hh = self.hh_scale(hh) if hasattr(self, 'hh_scale') else hh

        # 跨子带交互
        if self.cross_subband:
            pool = torch.cat([F.adaptive_avg_pool2d(lh,1), F.adaptive_avg_pool2d(hl,1), F.adaptive_avg_pool2d(hh,1)], dim=1)
            ll = ll * self.high_gate(pool)

        # 输出
        if self.keep_subbands:
            out_sub = self.c2 // 4
            ll_out = self.ll_proj(ll); lh_out = self.lh_proj(lh); hl_out = self.hl_proj(hl); hh_out = self.hh_proj(hh)
            out = torch.stack([ll_out, lh_out, hl_out, hh_out], dim=2).reshape(B, self.c2, H//2, W//2)
        else:
            fused = torch.stack([ll, lh, hl, hh], dim=2).reshape(B, C*4, H//2, W//2)
            out = self.fuse(fused)

        return out


# ============== 测试 ==============
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== 标准模式 ===")
    m1 = WaveletRefine(128, 256, n=2, texture_suppress=True, use_eca=True)
    m1.to(device)
    out1 = m1(torch.randn(2, 128, 40, 40, device=device))
    print(out1.shape)  # [2,256,20,20]

    print("\n=== keep_subbands ===")
    m2 = WaveletRefine(128, 256, n=2, keep_subbands=True, texture_suppress=True, use_eca=True)
    m2.to(device)
    out2 = m2(torch.randn(2, 128, 40, 40, device=device))
    print(out2.shape)

    print("\n=== decompose_ll_only ===")
    # 模拟上层输出：子带堆叠，4*64=256通道
    x_sub = torch.randn(2, 256, 40, 40, device=device)
    m3 = WaveletRefine(256, 256, n=2, decompose_ll_only=True, texture_suppress=False, high_scale=True, use_eca=True)
    m3.to(device)
    out3 = m3(x_sub)
    print(out3.shape)  # 应为 [2,256,20,20]