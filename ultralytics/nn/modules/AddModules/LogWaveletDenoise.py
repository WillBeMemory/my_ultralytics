import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pywt

from ultralytics.nn.modules.SPDConv import SPDConv


# 导入已有的 SPDConv 模块（请根据实际路径调整）
  # 假设 SPDConv 类定义在 spdconv.py 中

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

        # ---------- 下采样分支：使用 SPDConv ----------
        if self.downsample:
            # SPDConv 替代原来的 Conv+BN+SiLU，s=2 执行无损下采样，内部包含激活和BN
            self.spd_conv = SPDConv(c1, c2, k=kernel_size, s=2, act=True)
        else:
            self.spd_conv = nn.Identity()

    # ---------- 小波变换核心（内部自动对齐 dtype） ----------
    def _dwt(self, x):
        B, C, H, W = x.shape
        dec_f = self.dec_filters.to(x.dtype)
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
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            x_log = torch.log(torch.clamp(x, min=1e-6))
            rec_log = self._multilevel_denoise(x_log)
            out = torch.exp(rec_log)  # float32
            if self.downsample:
                # 对齐 SPDConv 权重的 dtype（通常为 float16）
                out = out.to(next(self.spd_conv.parameters()).dtype)
                out = self.spd_conv(out)
        return out.to(orig_dtype)


# ---------- 简单测试 ----------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 测试下采样模式（包含 SPDConv）
    model = LogWaveletDenoise(c1=64, c2=128, level=1, downsample=True, kernel_size=3).to(device)
    model.train()
    x = torch.randn(2, 64, 32, 32, dtype=torch.float16).to(device)
    y = model(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape} (expected [2,128,16,16])")
    loss = y.mean()
    loss.backward()
    print("Backward passed. Test OK!")

    # 测试不下采样模式
    model2 = LogWaveletDenoise(c1=32, c2=32, level=2, downsample=False).to(device)
    model2.train()
    x2 = torch.randn(1, 32, 20, 20, dtype=torch.float16).to(device)
    y2 = model2(x2)
    print(f"Input shape: {x2.shape}, Output shape: {y2.shape} (expected [1,32,20,20])")
    loss2 = y2.mean()
    loss2.backward()
    print("Backward passed. Test OK!")