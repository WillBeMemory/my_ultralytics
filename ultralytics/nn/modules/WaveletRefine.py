import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv, Bottleneck


# ---------- 高效通道注意力 ----------
class ECA(nn.Module):
    """Efficient Channel Attention"""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(dim=(2, 3), keepdim=True)
        y = y.squeeze(-1).transpose(-1, -2)          # (B, 1, C)
        y = self.conv(y)                             # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)        # (B, C, 1, 1)
        return x * self.sigmoid(y)


# ---------- 轻量星操作块 ----------
class StarBlock(nn.Module):
    """轻量星操作块，输入输出通道可以不同"""
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


# ---------- 小波纹理抑制 ----------
class WaveletTextureSuppress(nn.Module):
    """用 StarBlock 实现小波子带纹理抑制"""
    def __init__(self, c):
        super().__init__()
        self.star = StarBlock(c * 2, 1, k=3, e=0.5, shortcut=False)
        self.bias = nn.Parameter(torch.tensor(2.0))      # 初始接近 1（不抑制）
        self.sigmoid = nn.Sigmoid()

    def forward(self, lh, hl):
        combined = torch.cat([lh, hl], dim=1)
        logit = self.star(combined) + self.bias.view(1, -1, 1, 1)
        mask = self.sigmoid(logit)
        return lh * mask, hl * mask


# ====================== WaveletRefine 完整版 ======================
class WaveletRefine(nn.Module):
    """
    小波域特征精炼 + 下采样模块。

    支持三种模式：
    1. 标准模式：四子带分解 → 各子带处理 → 1×1 融合输出
    2. keep_subbands 模式：四子带分解 → 各子带处理 → 独立投影 → 子带堆叠输出
    3. decompose_ll_only 模式：仅对输入 LL 子带分解，历史高频经最大池化下采样后与新高频合并，
       保持频域分离，输出子带堆叠。

    参数:
        c1, c2              : 输入/输出通道数
        n, shortcut, g, k   : LL 精炼 Bottleneck 参数
        texture_suppress    : 是否对 LH/HL 使用 StarBlock 纹理抑制（否则高频 1×1 缩放）
        keep_subbands       : 输出是否保持子带堆叠（每个子带占 c2/4）
        use_eca             : LL 精炼后是否添加 ECA
        high_scale          : texture_suppress=False 时，是否对高频子带应用 1×1 缩放
        cross_subband       : 是否启用跨子带交互（高频统计量注入 LL）
        decompose_ll_only   : 是否仅分解 LL，复用历史高频（本模式下 keep_subbands 自动为 True）
    """
    def __init__(
        self,
        c1, c2,
        n=1, shortcut=True, g=1, k=3,
        texture_suppress=True,
        keep_subbands=False,
        use_eca=True,
        high_scale=True,
        cross_subband=False,
        decompose_ll_only=False,
    ):
        super().__init__()
        assert c2 % 4 == 0, "c2 must be divisible by 4 for subband split"
        self.c1, self.c2 = c1, c2
        self.texture_suppress = texture_suppress
        self.high_scale = high_scale if not texture_suppress else False
        self.keep_subbands = keep_subbands
        self.cross_subband = cross_subband
        self.decompose_ll_only = decompose_ll_only

        # ---- Haar 分解滤波器 ----
        w_ll = torch.tensor([[1., 1.], [1., 1.]]) / 2.0
        w_lh = torch.tensor([[1., -1.], [1., -1.]]) / 2.0
        w_hl = torch.tensor([[1., 1.], [-1., -1.]]) / 2.0
        w_hh = torch.tensor([[1., -1.], [-1., 1.]]) / 2.0
        haar_init = torch.stack([w_ll, w_lh, w_hl, w_hh], dim=0).unsqueeze(1)  # (4,1,2,2)

        # 根据不同模式构建特有的子模块
        if decompose_ll_only:
            # 输入是四子带堆叠，每个子带占 c1//4 通道
            sub_c = c1 // 4
            self.register_buffer('haar_ll_filter', haar_init.repeat(sub_c, 1, 1, 1))

            # LL 精炼（在子带通道上操作）
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

            # 历史高频下采样：无参最大池化
            self.downsample_high = nn.MaxPool2d(kernel_size=2, stride=2)

            # 子带合并与投影
            out_sub = c2 // 4
            self.ll_proj = Conv(sub_c, out_sub, k=1, act=False)
            self.lh_combiner = Conv(sub_c * 2, out_sub, k=1, act=False)
            self.hl_combiner = Conv(sub_c * 2, out_sub, k=1, act=False)
            self.hh_combiner = Conv(sub_c * 2, out_sub, k=1, act=False)

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

        # 跨子带交互（仅在常规模式且 cross_subband=True 时有效）
        if cross_subband and not decompose_ll_only:
            self.high_gate = nn.Sequential(
                nn.Conv2d(c1 * 3, c1, 1, bias=False),
                nn.Sigmoid()
            )

        # ---- 输出投影 ----
        out_sub = c2 // 4
        if decompose_ll_only:
            # 已在上面构建，这里什么都不做
            pass
        elif keep_subbands:
            self.ll_proj = Conv(c1, out_sub, k=1, act=False)
            self.lh_proj = Conv(c1, out_sub, k=1, act=False)
            self.hl_proj = Conv(c1, out_sub, k=1, act=False)
            self.hh_proj = Conv(c1, out_sub, k=1, act=False)
        else:
            self.fuse = Conv(c1 * 4, c2, k=1, act=True)

    # ==================== 前向传播 ====================
    def forward(self, x):
        B, C, H, W = x.shape

        # ---------- 模式 3：decompose_ll_only ----------
        if self.decompose_ll_only:
            sub_c = self.c1 // 4
            # 分离四个子带
            ll1 = x[:, :sub_c]
            lh1 = x[:, sub_c:2*sub_c]
            hl1 = x[:, 2*sub_c:3*sub_c]
            hh1 = x[:, 3*sub_c:]

            # 仅对 LL1 进行 Haar 分解
            coeff = F.conv2d(ll1, self.haar_ll_filter, stride=2, groups=sub_c)
            coeff = coeff.view(B, sub_c, 4, H//2, W//2)
            ll2 = coeff[:, :, 0]; lh2 = coeff[:, :, 1]; hl2 = coeff[:, :, 2]; hh2 = coeff[:, :, 3]

            # LL2 精炼 + ECA
            ll2 = self.ll_refine_sub(ll2)
            ll2 = self.ll_eca_sub(ll2)

            # 高频处理（新分解的 LH2/HL2/HH2）
            if self.texture_suppress:
                lh2, hl2 = self.texture_module_sub(lh2, hl2)
            elif self.high_scale:
                lh2 = self.lh_scale_sub(lh2)
                hl2 = self.hl_scale_sub(hl2)
                hh2 = self.hh_scale_sub(hh2) if hasattr(self, 'hh_scale_sub') else hh2

            # 历史高频下采样（最大池化）
            lh1_down = self.downsample_high(lh1)
            hl1_down = self.downsample_high(hl1)
            hh1_down = self.downsample_high(hh1)

            # 合并新旧高频，压缩到 c2/4
            ll_out = self.ll_proj(ll2)
            lh_out = self.lh_combiner(torch.cat([lh2, lh1_down], dim=1))
            hl_out = self.hl_combiner(torch.cat([hl2, hl1_down], dim=1))
            hh_out = self.hh_combiner(torch.cat([hh2, hh1_down], dim=1))

            # 堆叠输出
            out = torch.stack([ll_out, lh_out, hl_out, hh_out], dim=2).reshape(B, self.c2, H//2, W//2)
            return out

        # ---------- 常规全分解模式 ----------
        # 补边至偶数
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # Haar 分解
        coeff = F.conv2d(x, self.haar_filter, stride=2, groups=C)
        coeff = coeff.view(B, C, 4, H//2, W//2)
        ll, lh, hl, hh = coeff[:, :, 0], coeff[:, :, 1], coeff[:, :, 2], coeff[:, :, 3]

        # LL 精炼 + ECA
        ll = self.ll_refine(ll)
        ll = self.ll_eca(ll)

        # 高频处理
        if self.texture_suppress:
            lh, hl = self.texture_module(lh, hl)
        elif self.high_scale:
            lh = self.lh_scale(lh); hl = self.hl_scale(hl)
            hh = self.hh_scale(hh) if hasattr(self, 'hh_scale') else hh

        # 跨子带交互（可选）
        if self.cross_subband and not self.decompose_ll_only:
            pool = torch.cat([
                F.adaptive_avg_pool2d(lh, 1),
                F.adaptive_avg_pool2d(hl, 1),
                F.adaptive_avg_pool2d(hh, 1)
            ], dim=1)
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


# ====================== 简单测试 ======================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== 标准模式 ===")
    m1 = WaveletRefine(128, 256, n=2, texture_suppress=True, use_eca=True)
    m1.to(device)
    out1 = m1(torch.randn(2, 128, 40, 40, device=device))
    print(out1.shape)  # [2,256,20,20]

    print("\n=== keep_subbands 模式 ===")
    m2 = WaveletRefine(128, 256, n=2, keep_subbands=True, texture_suppress=True, use_eca=True)
    m2.to(device)
    out2 = m2(torch.randn(2, 128, 40, 40, device=device))
    print(out2.shape)  # [2,256,20,20]

    print("\n=== decompose_ll_only 模式 (最大池化下采样) ===")
    # 模拟上层输出：子带堆叠，4*64=256 通道
    x_sub = torch.randn(2, 256, 40, 40, device=device)
    m3 = WaveletRefine(256, 256, n=2, decompose_ll_only=True, texture_suppress=False, high_scale=True, use_eca=True)
    m3.to(device)
    out3 = m3(x_sub)
    print(out3.shape)  # [2,256,20,20]

    print("\n所有测试通过！")