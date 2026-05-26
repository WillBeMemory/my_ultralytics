import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pywt

class LogWaveletDenoise(nn.Module):
    def __init__(self, c1, c2, level=1, downsample=True, kernel_size=3, wavelet='bior4.4'):
        super().__init__()
        self.downsample = downsample
        self.level = level

        # ---------- 固定小波基 (bior4.4) ----------
        w = pywt.Wavelet(wavelet)
        dec_lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)
        dec_hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)
        rec_lo = torch.tensor(w.rec_lo, dtype=torch.float32)
        rec_hi = torch.tensor(w.rec_hi, dtype=torch.float32)

        dec_LL = (dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        dec_LH = (dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        dec_HL = (dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        dec_HH = (dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        self.register_buffer('dec_filters', torch.cat([dec_LL, dec_LH, dec_HL, dec_HH], dim=0))

        rec_LL = (rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        rec_LH = (rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        rec_HL = (rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        rec_HH = (rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        self.register_buffer('rec_filters', torch.cat([rec_LL, rec_LH, rec_HL, rec_HH], dim=0))

        # ---------- 可学习阈值 ----------
        self.log_thresh_LH = nn.ParameterList([
            nn.Parameter(torch.full((1, c1, 1, 1), -2.0)) for _ in range(level)
        ])
        self.log_thresh_HL = nn.ParameterList([
            nn.Parameter(torch.full((1, c1, 1, 1), -2.0)) for _ in range(level)
        ])
        self.log_thresh_HH = nn.ParameterList([
            nn.Parameter(torch.full((1, c1, 1, 1), -2.0)) for _ in range(level)
        ])

        # ---------- 下采样分支 ----------
        if self.downsample:
            self.down_conv = nn.Conv2d(c1, c2, kernel_size, stride=2,
                                       padding=kernel_size // 2, bias=False)
            self.down_bn = nn.BatchNorm2d(c2)
            self.down_act = nn.SiLU(inplace=True)

    # ---------- 小波变换核心（内部自动对齐 dtype） ----------
    def _dwt(self, x):
        B, C, H, W = x.shape
        dec_f = self.dec_filters.to(x.dtype)  # 关键：对齐滤波器与输入精度
        pad_len = dec_f.shape[-1] // 2
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        x = F.pad(x, (pad_len, pad_len, pad_len, pad_len), mode='reflect')
        filters = dec_f.repeat(C, 1, 1, 1)
        coeffs = F.conv2d(x, filters, stride=2, groups=C)
        coeffs = coeffs.view(B, C, 4, coeffs.shape[-2], coeffs.shape[-1])
        return coeffs[:, :, 0], coeffs[:, :, 1], coeffs[:, :, 2], coeffs[:, :, 3]

    def _idwt(self, LL, LH, HL, HH, out_h, out_w):
        rec_f = self.rec_filters.to(LL.dtype)
        B, C, H, W = LL.shape
        coeffs = torch.stack([LL, LH, HL, HH], dim=2).reshape(B, C * 4, H, W)
        pad_len = rec_f.shape[-1] // 2
        filters = rec_f.repeat(C, 1, 1, 1)
        out = F.conv_transpose2d(coeffs, filters, stride=2, padding=pad_len, groups=C)
        return out[:, :, :out_h, :out_w]

    def _soft_threshold(self, coeff, lambd):
        return torch.where(
            coeff > lambd, coeff - lambd,
            torch.where(coeff < -lambd, coeff + lambd, torch.zeros_like(coeff))
        )

    def _multilevel_denoise(self, x_log):
        detail_coeffs = []
        current = x_log
        for lvl in range(self.level):
            in_h, in_w = current.shape[2], current.shape[3]
            LL, LH, HL, HH = self._dwt(current)
            detail_coeffs.append((LH, HL, HH, in_h, in_w))
            current = LL

        for lvl in reversed(range(self.level)):
            LH, HL, HH, in_h, in_w = detail_coeffs[lvl]
            # 阈值也显式转为 float32
            tau_lh = F.softplus(self.log_thresh_LH[lvl].float())
            tau_hl = F.softplus(self.log_thresh_HL[lvl].float())
            tau_hh = F.softplus(self.log_thresh_HH[lvl].float())
            LH = self._soft_threshold(LH, tau_lh)
            HL = self._soft_threshold(HL, tau_hl)
            HH = self._soft_threshold(HH, tau_hh)
            current = self._idwt(current, LH, HL, HH, in_h, in_w)
        return current

    def forward(self, x):
        orig_dtype = x.dtype
        # 1. 小波变换全程在 float32 下运行
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            x_log = torch.log(torch.clamp(x, min=1e-6))
            rec_log = self._multilevel_denoise(x_log)
            out = torch.exp(rec_log)  # float32

            # 2. 下采样分支：将输出转为权重类型（通常是 half），避免类型不匹配
            if self.downsample:
                out = out.to(self.down_conv.weight.dtype)
                out = self.down_act(self.down_bn(self.down_conv(out)))

        # 3. 最终输出与原始输入类型一致
        return out.to(orig_dtype)