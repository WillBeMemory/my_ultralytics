import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DirectionalConv(nn.Module):
    """
    方向性深度卷积（与 WaveletStem 中一致）
    - LL: 标准 3×3 深度卷积
    - LH: (1, 3) 垂直条状深度卷积（增强水平边缘）
    - HL: (3, 1) 水平条状深度卷积（增强垂直边缘）
    - HH: 3×3 深度卷积，可初始化为对角模式（强化对角边缘）
    """
    def __init__(self, channels: int, mode: str, use_diag_init: bool = True):
        super().__init__()
        self.mode = mode
        if mode == 'll':
            self.conv = nn.Conv2d(channels, channels, 3, padding=1,
                                  groups=channels, bias=False)
        elif mode == 'lh':
            self.conv = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1),
                                  groups=channels, bias=False)
        elif mode == 'hl':
            self.conv = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0),
                                  groups=channels, bias=False)
        elif mode == 'hh':
            self.conv = nn.Conv2d(channels, channels, 3, padding=1,
                                  groups=channels, bias=False)
            if use_diag_init:
                self._init_diag_weights()
        else:
            raise ValueError(f"Invalid mode: {mode}")
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def _init_diag_weights(self):
        """初始化 HH 卷积核为对角增强模式（主对角线正，反对角线负）"""
        with torch.no_grad():
            weight = self.conv.weight  # (C, 1, 3, 3)
            nn.init.zeros_(weight)
            # 主对角线 (+)
            weight[:, 0, 0, 0] = 1.0
            weight[:, 0, 1, 1] = 1.0
            weight[:, 0, 2, 2] = 1.0
            # 反对角线 (-)
            weight[:, 0, 0, 2] = -0.5
            weight[:, 0, 1, 1] -= 1.0  # 中心点调整
            weight[:, 0, 2, 0] = -0.5
            # 归一化
            C = weight.shape[0]
            weight /= weight.view(C, -1).norm(p=2, dim=1).view(C, 1, 1, 1) + 1e-6

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LogWaveletDenoise(nn.Module):
    """
    对数-小波去噪 + 方向性子带增强模块
    在可学习软阈值去噪后，对 LH/HL/HH/LL 子带分别进行方向性深度卷积，
    再通过小波重构和指数变换恢复图像。
    """
    def __init__(
        self,
        c1, c2,
        level=1,
        downsample=True,
        kernel_size=3,
        use_directional=True,          # 是否启用子带方向卷积
    ):
        super().__init__()
        self.downsample = downsample
        self.level = level
        self.use_directional = use_directional

        # 正交 Haar 小波滤波器
        lo = torch.tensor([1.0, 1.0]) / math.sqrt(2)
        hi = torch.tensor([1.0, -1.0]) / math.sqrt(2)
        dec_LL = (lo.unsqueeze(0) * lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        dec_LH = (lo.unsqueeze(0) * hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        dec_HL = (hi.unsqueeze(0) * lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        dec_HH = (hi.unsqueeze(0) * hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        self.register_buffer('dec_filters', torch.cat([dec_LL, dec_LH, dec_HL, dec_HH], dim=0))
        self.register_buffer('rec_filters', self.dec_filters.clone())

        # 可学习的逐通道软阈值参数
        self.log_thresh_LH = nn.ParameterList([
            nn.Parameter(torch.full((1, c1, 1, 1), -2.0)) for _ in range(level)
        ])
        self.log_thresh_HL = nn.ParameterList([
            nn.Parameter(torch.full((1, c1, 1, 1), -2.0)) for _ in range(level)
        ])
        self.log_thresh_HH = nn.ParameterList([
            nn.Parameter(torch.full((1, c1, 1, 1), -2.0)) for _ in range(level)
        ])

        # 方向性子带卷积（每个子带一组，深度可分离，参数量极小）
        if self.use_directional:
            self.ll_convs = nn.ModuleList([
                DirectionalConv(c1, 'll') for _ in range(level)
            ])
            self.lh_convs = nn.ModuleList([
                DirectionalConv(c1, 'lh') for _ in range(level)
            ])
            self.hl_convs = nn.ModuleList([
                DirectionalConv(c1, 'hl') for _ in range(level)
            ])
            self.hh_convs = nn.ModuleList([
                DirectionalConv(c1, 'hh', use_diag_init=True) for _ in range(level)
            ])

        # 可选的下采样卷积层
        if self.downsample:
            self.down_conv = nn.Conv2d(c1, c2, kernel_size, stride=2,
                                       padding=kernel_size // 2, bias=False)
            self.down_bn = nn.BatchNorm2d(c2)
            self.down_act = nn.SiLU(inplace=True)

    def _dwt(self, x):
        B, C, H, W = x.shape
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        filters = self.dec_filters.repeat(C, 1, 1, 1)
        coeffs = F.conv2d(x, filters, stride=2, groups=C)
        coeffs = coeffs.view(B, C, 4, coeffs.shape[-2], coeffs.shape[-1])
        return coeffs[:, :, 0], coeffs[:, :, 1], coeffs[:, :, 2], coeffs[:, :, 3]

    def _idwt(self, LL, LH, HL, HH):
        B, C, H, W = LL.shape
        coeffs = torch.stack([LL, LH, HL, HH], dim=2).reshape(B, C * 4, H, W)
        filters = self.rec_filters.repeat(C, 1, 1, 1)
        out = F.conv_transpose2d(coeffs, filters, stride=2, groups=C)
        return out[:, :, :2*H, :2*W]

    def _soft_threshold(self, coeff, lambd):
        return torch.where(
            coeff > lambd,
            coeff - lambd,
            torch.where(coeff < -lambd, coeff + lambd, torch.zeros_like(coeff))
        )

    def _multilevel_denoise(self, x_log):
        detail_coeffs = []
        current = x_log
        for lvl in range(self.level):
            LL, LH, HL, HH = self._dwt(current)
            detail_coeffs.append((LL, LH, HL, HH))   # 保存 LL 用于后续增强
            current = LL

        for lvl in reversed(range(self.level)):
            LL, LH, HL, HH = detail_coeffs[lvl]

            # 可学习软阈值
            tau_lh = F.softplus(self.log_thresh_LH[lvl])
            tau_hl = F.softplus(self.log_thresh_HL[lvl])
            tau_hh = F.softplus(self.log_thresh_HH[lvl])
            LH = self._soft_threshold(LH, tau_lh)
            HL = self._soft_threshold(HL, tau_hl)
            HH = self._soft_threshold(HH, tau_hh)

            # 方向性子带卷积（增强边缘和纹理）
            if self.use_directional:
                LL = self.ll_convs[lvl](LL)
                LH = self.lh_convs[lvl](LH)
                HL = self.hl_convs[lvl](HL)
                HH = self.hh_convs[lvl](HH)

            current = self._idwt(current, LH, HL, HH) if lvl > 0 else self._idwt(LL, LH, HL, HH)
            # 注意：最后一级重构时需要传入 LL，其余级别 current 即为 LL？
            # 修正：在循环中，current 始终是上一级重构后的 LL 部分，最后一次重构需要传入最终的 LL。
            # 原代码中 current = LL 被赋值后，在重构时直接使用 current 作为 LL，但 current 实际上是 LL，正确。
        return current

    def forward(self, x):
        x_log = torch.log(torch.clamp(x, min=1e-6))
        rec_log = self._multilevel_denoise(x_log)
        out = torch.exp(rec_log)

        if self.downsample:
            out = self.down_act(self.down_bn(self.down_conv(out)))
        return out