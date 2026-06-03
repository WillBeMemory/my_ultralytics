import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LogWaveletDenoise(nn.Module):
    def __init__(self, c1, c2, level=1, downsample=True, kernel_size=3):
        super().__init__()
        self.downsample = downsample
        self.level = level

        # ---------- Haar 小波滤波器 ----------
        lo = torch.tensor([1.0, 1.0]) / math.sqrt(2)
        hi = torch.tensor([1.0, -1.0]) / math.sqrt(2)
        dec_LL = (lo.unsqueeze(0) * lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        dec_LH = (lo.unsqueeze(0) * hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        dec_HL = (hi.unsqueeze(0) * lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        dec_HH = (hi.unsqueeze(0) * hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        self.register_buffer('dec_filters', torch.cat([dec_LL, dec_LH, dec_HL, dec_HH], dim=0))
        self.register_buffer('rec_filters', self.dec_filters.clone())

        # ---------- 可学习阈值缩放因子 ----------
        self.thresh_scale_LH = nn.ParameterList([
            nn.Parameter(torch.full((1, c1, 1, 1), 0.0)) for _ in range(level)
        ])
        self.thresh_scale_HL = nn.ParameterList([
            nn.Parameter(torch.full((1, c1, 1, 1), 0.0)) for _ in range(level)
        ])
        self.thresh_scale_HH = nn.ParameterList([
            nn.Parameter(torch.full((1, c1, 1, 1), 0.0)) for _ in range(level)
        ])

        # ---------- 噪声估计基值 ----------
        self.register_buffer('noise_sigma', torch.ones(level, c1) * 0.1)
        self.register_buffer('noise_initialized', torch.tensor(False))

        # ---------- 可选下采样 ----------
        if self.downsample:
            self.down_conv = nn.Conv2d(c1, c2, kernel_size, stride=2,
                                       padding=kernel_size // 2, bias=False)
            self.down_bn = nn.BatchNorm2d(c2)
            self.down_act = nn.SiLU(inplace=True)

    def _dwt(self, x):
        B, C, H, W = x.shape
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        # 确保滤波器与输入类型一致（兼容半精度模型）
        filters = self.dec_filters.to(x.dtype).repeat(C, 1, 1, 1)
        coeffs = F.conv2d(x, filters, stride=2, groups=C)
        coeffs = coeffs.view(B, C, 4, coeffs.shape[-2], coeffs.shape[-1])
        return coeffs[:, :, 0], coeffs[:, :, 1], coeffs[:, :, 2], coeffs[:, :, 3]

    def _idwt(self, LL, LH, HL, HH, orig_h, orig_w):
        B, C, H, W = LL.shape
        coeffs = torch.stack([LL, LH, HL, HH], dim=2).reshape(B, C * 4, H, W)
        # 重构滤波器同样类型对齐
        filters = self.rec_filters.to(LL.dtype).repeat(C, 1, 1, 1)
        out = F.conv_transpose2d(coeffs, filters, stride=2, groups=C)
        return out[:, :, :orig_h, :orig_w]

    def _soft_threshold(self, coeff, lambd):
        return torch.where(
            coeff > lambd, coeff - lambd,
            torch.where(coeff < -lambd, coeff + lambd, torch.zeros_like(coeff))
        )

    def _estimate_noise_base(self, HH):
        hh_abs = torch.abs(HH.reshape(HH.size(0), HH.size(1), -1))
        sorted_abs = torch.sort(hh_abs, dim=-1)[0]
        N = hh_abs.shape[-1]
        k = N // 2
        sigma = sorted_abs[:, :, k] / 0.6745
        return sigma * math.sqrt(2.0 * math.log(N + 1e-6))  # (B, C)

    def _multilevel_denoise(self, x_log):
        B, C, H, W = x_log.shape
        detail_coeffs = []
        current = x_log
        for lvl in range(self.level):
            cur_h, cur_w = current.shape[2], current.shape[3]
            LL, LH, HL, HH = self._dwt(current)
            detail_coeffs.append((LH, HL, HH, cur_h, cur_w))
            current = LL

        for lvl in reversed(range(self.level)):
            LH, HL, HH, orig_h, orig_w = detail_coeffs[lvl]

            # 阈值参数和噪声基值统一 float32
            scale_lh = F.softplus(self.thresh_scale_LH[lvl].float())
            scale_hl = F.softplus(self.thresh_scale_HL[lvl].float())
            scale_hh = F.softplus(self.thresh_scale_HH[lvl].float())
            noise = self.noise_sigma[lvl].float().view(1, -1, 1, 1)

            tau_lh = scale_lh * noise
            tau_hl = scale_hl * noise
            tau_hh = scale_hh * noise

            LH = self._soft_threshold(LH, tau_lh)
            HL = self._soft_threshold(HL, tau_hl)
            HH = self._soft_threshold(HH, tau_hh)

            current = self._idwt(current, LH, HL, HH, orig_h, orig_w)

        if current.shape[2] != H or current.shape[3] != W:
            current = current[:, :, :H, :W]
        return current

    def forward(self, x):
        orig_dtype = x.dtype
        # 全程 float32 运算，完全不受外部 AMP 影响
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            x_clamped = torch.clamp(x, min=1e-6)
            x_log = torch.log(x_clamped)

            # 首次训练前向时初始化噪声基值
            if self.training and not self.noise_initialized:
                with torch.no_grad():
                    _, _, _, HH = self._dwt(x_log)
                    noise_base = self._estimate_noise_base(HH).mean(dim=0)  # (C,)
                    self.noise_sigma.copy_(noise_base.unsqueeze(0).expand(self.level, -1).float())
                    self.noise_initialized.copy_(torch.tensor(True))

            rec_log = self._multilevel_denoise(x_log)
            rec_log = torch.clamp(rec_log, max=20.0)
            out = torch.exp(rec_log)

            # 残差融合
            if out.shape == x.shape:
                out = 0.5 * out + 0.5 * x

            if self.downsample:
                # 下采样分支：匹配卷积权重类型
                out = out.to(self.down_conv.weight.dtype)
                out = self.down_act(self.down_bn(self.down_conv(out)))

        return out.to(orig_dtype)