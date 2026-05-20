import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LogWaveletDenoise(nn.Module):
    def __init__(self, c1, c2, level=1, downsample=True, kernel_size=3):
        super().__init__()
        self.downsample = downsample
        self.level = level

        # 正交 Haar 小波滤波器（不变）
        lo = torch.tensor([1.0, 1.0]) / math.sqrt(2)
        hi = torch.tensor([1.0, -1.0]) / math.sqrt(2)
        dec_LL = (lo.unsqueeze(0) * lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        dec_LH = (lo.unsqueeze(0) * hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        dec_HL = (hi.unsqueeze(0) * lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        dec_HH = (hi.unsqueeze(0) * hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        self.register_buffer('dec_filters', torch.cat([dec_LL, dec_LH, dec_HL, dec_HH], dim=0))
        self.register_buffer('rec_filters', self.dec_filters.clone())

        # 可学习的逐通道阈值参数（为每个细节子带每级各维护一组）
        # 用 softplus 保证阈值始终为正
        self.log_thresh_LH = nn.ParameterList([
            nn.Parameter(torch.full((1, c1, 1, 1), -2.0))  # 初始 softplus(-2) ≈ 0.14，很小
            for _ in range(level)
        ])
        self.log_thresh_HL = nn.ParameterList([
            nn.Parameter(torch.full((1, c1, 1, 1), -2.0)) for _ in range(level)
        ])
        self.log_thresh_HH = nn.ParameterList([
            nn.Parameter(torch.full((1, c1, 1, 1), -2.0)) for _ in range(level)
        ])

        # 可选的下采样卷积层
        if self.downsample:
            self.down_conv = nn.Conv2d(c1, c2, kernel_size, stride=2, padding=kernel_size // 2, bias=False)
            self.down_bn = nn.BatchNorm2d(c2)
            self.down_act = nn.SiLU(inplace=True)

    # --- 小波分解、重构、软阈值函数（不变）---
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
        """可微软阈值，lambd 形状为 (1, C, 1, 1) 可广播"""
        return torch.where(
            coeff > lambd,
            coeff - lambd,
            torch.where(coeff < -lambd, coeff + lambd, torch.zeros_like(coeff))
        )

    # --- 多级去噪核心（使用可学习阈值）---
    def _multilevel_denoise(self, x_log):
        detail_coeffs = []
        current = x_log
        for lvl in range(self.level):
            LL, LH, HL, HH = self._dwt(current)
            detail_coeffs.append((LH, HL, HH))
            current = LL

        # 逐级去噪并重构
        for lvl in reversed(range(self.level)):
            LH, HL, HH = detail_coeffs[lvl]
            # 获取当前级别的可学习阈值（通过 softplus 保证正数）
            tau_lh = F.softplus(self.log_thresh_LH[lvl])
            tau_hl = F.softplus(self.log_thresh_HL[lvl])
            tau_hh = F.softplus(self.log_thresh_HH[lvl])

            # 分别施加软阈值
            LH = self._soft_threshold(LH, tau_lh)
            HL = self._soft_threshold(HL, tau_hl)
            HH = self._soft_threshold(HH, tau_hh)

            current = self._idwt(current, LH, HL, HH)
        return current

    def forward(self, x):
        x_log = torch.log(torch.clamp(x, min=1e-6))
        rec_log = self._multilevel_denoise(x_log)
        out = torch.exp(rec_log)

        if self.downsample:
            out = self.down_act(self.down_bn(self.down_conv(out)))
        return out