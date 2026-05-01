import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv, Bottleneck


class StarBlock(nn.Module):
    """轻量星操作块，输入输出通道可以不同"""
    def __init__(self, c1, c2, k=3, e=0.5, shortcut=False, act=nn.ReLU6(inplace=True)):
        super().__init__()
        hidden = max(1, int(c1 * e))
        self.hidden = hidden
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
        x1 = self.dw1(x)
        x1 = self.pw1(x1)
        x1 = self.bn1(x1)
        x1 = self.act1(x1)

        x2 = self.dw2(x)
        x2 = self.pw2(x2)
        x2 = self.bn2(x2)

        out = x1 * x2                     # 星操作
        out = self.fusion(out)
        out = self.out_act(out)
        if self.add:
            out = out + x
        return out


class WaveletTextureSuppress(nn.Module):
    """用 StarBlock 实现小波子带纹理抑制"""
    def __init__(self, c, reduction=4):
        super().__init__()
        self.star = StarBlock(c * 2, 1, k=3, e=0.5, shortcut=False)
        self.bias = nn.Parameter(torch.tensor(2.0))
        self.sigmoid = nn.Sigmoid()

    def forward(self, lh, hl):
        combined = torch.cat([lh, hl], dim=1)
        logit = self.star(combined) + self.bias.view(1, -1, 1, 1)
        mask = self.sigmoid(logit)
        lh = lh * mask
        hl = hl * mask
        return lh, hl


class WaveletRefine(nn.Module):
    """
    小波域特征精炼 + 下采样模块（子带堆叠输出）。
    recursive=False: 对完整的输入特征进行 Haar 分解。
    recursive=True:  假设输入是前一级输出的子带堆叠，仅取其前 1/4 通道（LL 部分）进行 Haar 分解。
    两种模式均输出子带堆叠格式，通道数为 c2（c2 需能被 4 整除）。
    """
    def __init__(self, c1, c2, n=1,recursive=False, shortcut=True, g=1, k=3,
                 texture_suppress=True ):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.texture_suppress = texture_suppress
        self.recursive = recursive

        # 实际用于 Haar 分解的输入通道数
        if recursive:
            self.decomp_ch = c1 // 4   # 取 LL 子带
        else:
            self.decomp_ch = c1

        # 固定 Haar 分解滤波器
        w_ll = torch.tensor([[1., 1.], [1., 1.]]) / 2.0
        w_lh = torch.tensor([[1., -1.], [1., -1.]]) / 2.0
        w_hl = torch.tensor([[1., 1.], [-1., -1.]]) / 2.0
        w_hh = torch.tensor([[1., -1.], [-1., 1.]]) / 2.0
        haar_init = torch.stack([w_ll, w_lh, w_hl, w_hh], dim=0).unsqueeze(1)
        self.register_buffer('haar_filter', haar_init.repeat(self.decomp_ch, 1, 1, 1))

        # LL 子带精炼分支
        self.ll_refine = nn.Sequential(*[
            Bottleneck(self.decomp_ch, self.decomp_ch, shortcut, g, k=(k, k), e=1.0) for _ in range(n)
        ])

        # 纹理抑制
        if texture_suppress:
            self.texture_module = WaveletTextureSuppress(self.decomp_ch)

        # 子带独立投影卷积（输出通道各占 c2/4）
        out_sub_ch = c2 // 4
        self.ll_proj = Conv(self.decomp_ch, out_sub_ch, k=1, act=False)
        self.lh_proj = Conv(self.decomp_ch, out_sub_ch, k=1, act=False)
        self.hl_proj = Conv(self.decomp_ch, out_sub_ch, k=1, act=False)
        self.hh_proj = Conv(self.decomp_ch, out_sub_ch, k=1, act=False)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 获取进行 Haar 分解的特征部分
        if self.recursive:
            # 输入是子带堆叠，取前 1/4 通道作为 LL
            ll_part = x[:, :self.decomp_ch]          # (B, c1//4, H, W)
            x = ll_part
        # 否则，x 就是普通特征图，直接使用

        # 2. 处理奇数尺寸（反射填充）
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # 3. Haar 分解
        coeff = F.conv2d(x, self.haar_filter, stride=2, groups=self.decomp_ch)  # (B, 4*decomp_ch, H/2, W/2)
        coeff = coeff.view(B, self.decomp_ch, 4, H // 2, W // 2)

        ll = coeff[:, :, 0]
        lh = coeff[:, :, 1]
        hl = coeff[:, :, 2]
        hh = coeff[:, :, 3]

        # 4. LL 精炼
        ll_out = self.ll_refine(ll)

        # 5. 纹理抑制
        if self.texture_suppress:
            lh_out, hl_out = self.texture_module(lh, hl)
        else:
            lh_out, hl_out = lh, hl

        hh_out = hh

        # 6. 各子带投影到输出通道
        ll_out = self.ll_proj(ll_out)
        lh_out = self.lh_proj(lh_out)
        hl_out = self.hl_proj(hl_out)
        hh_out = self.hh_proj(hh_out)

        # 7. 堆叠为子带格式 (B, c2, H/2, W/2)
        out = torch.stack([ll_out, lh_out, hl_out, hh_out], dim=2).reshape(B, self.c2, H // 2, W // 2)
        return out


# ==================== 测试 ====================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 正常模式测试 (c1=64, c2=128)
    x = torch.randn(2, 64, 80, 80).to(device)
    refine_normal = WaveletRefine(64, 128, recursive=False).to(device)
    out = refine_normal(x)
    print(f"Normal mode: input {x.shape} -> output {out.shape}")
    assert out.shape == (2, 128, 40, 40)

    # 递归模式测试：输入是子带堆叠 (c1=128，取前32通道作为LL)
    x_rec = torch.randn(2, 128, 40, 40).to(device)
    refine_rec = WaveletRefine(128, 256, recursive=True).to(device)
    out_rec = refine_rec(x_rec)
    print(f"Recursive mode: input {x_rec.shape} -> output {out_rec.shape}")
    assert out_rec.shape == (2, 256, 20, 20)

    # 参数对比
    params_normal = sum(p.numel() for p in refine_normal.parameters())
    params_rec = sum(p.numel() for p in refine_rec.parameters())
    print(f"Normal params: {params_normal:,}, Recursive params: {params_rec:,}")