import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LogWaveletDenoise(nn.Module):
    """
    同态小波去噪 + 可选卷积下采样 (YOLO 风格接口)
    """
    def __init__(self, c1, c2, threshold_factor=0.5, level=1, downsample=True, kernel_size=3):
        super().__init__()
        self.downsample = downsample
        self.level = level
        self.threshold_factor = threshold_factor

        # 正交 Haar 小波滤波器
        lo = torch.tensor([1.0, 1.0]) / math.sqrt(2)
        hi = torch.tensor([1.0, -1.0]) / math.sqrt(2)
        dec_LL = (lo.unsqueeze(0) * lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        dec_LH = (lo.unsqueeze(0) * hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        dec_HL = (hi.unsqueeze(0) * lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        dec_HH = (hi.unsqueeze(0) * hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)

        self.register_buffer('dec_filters', torch.cat([dec_LL, dec_LH, dec_HL, dec_HH], dim=0))
        self.register_buffer('rec_filters', self.dec_filters.clone())

        if self.downsample:
            self.down_conv = nn.Conv2d(c1, c2, kernel_size, stride=2, padding=kernel_size//2, bias=False)
            self.down_bn   = nn.BatchNorm2d(c2)
            self.down_act  = nn.SiLU(inplace=True)

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

    def _median_deterministic(self, x):
        """确定性中值：排序取中间值（替代 torch.median）"""
        sorted_x = torch.sort(x, dim=1)[0]
        k = x.shape[1] // 2
        if x.shape[1] % 2 == 1:
            return sorted_x[:, k]
        else:
            return (sorted_x[:, k-1] + sorted_x[:, k]) / 2.0

    def _estimate_noise_sigma(self, HH):
        hh_abs = torch.abs(HH.reshape(HH.shape[0], -1))
        median = self._median_deterministic(hh_abs)
        return median / 0.6745

    def _soft_threshold(self, coeff, lambd):
        return torch.sign(coeff) * torch.clamp(torch.abs(coeff) - lambd, min=0)

    def _multilevel_denoise(self, x_log):
        detail_coeffs = []
        current = x_log
        for lvl in range(self.level):
            LL, LH, HL, HH = self._dwt(current)
            detail_coeffs.append((LH, HL, HH))
            current = LL

        sigma = self._estimate_noise_sigma(detail_coeffs[0][2])

        for lvl in reversed(range(self.level)):
            LH, HL, HH = detail_coeffs[lvl]
            N = LH.numel()
            lambd = sigma * math.sqrt(2.0 * math.log(N)) * self.threshold_factor
            lambd = lambd.view(-1, 1, 1, 1)
            LH = self._soft_threshold(LH, lambd)
            HL = self._soft_threshold(HL, lambd)
            HH = self._soft_threshold(HH, lambd)
            current = self._idwt(current, LH, HL, HH)
        return current

    def forward(self, x):
        x_log = torch.log(x + 1e-10)
        rec_log = self._multilevel_denoise(x_log)
        out = torch.exp(rec_log)

        if self.downsample:
            out = self.down_act(self.down_bn(self.down_conv(out)))
        return out

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.rand(2, 3, 640, 640, device=device)  # RGB 输入

    # 下采样模式
    m = LogWaveletDenoise(c1=3, c2=32, threshold_factor=0.3, level=1, downsample=True).to(device)
    y = m(x)
    print(f"Downsample mode: {x.shape} → {y.shape}")   # [2,3,640,640] → [2,32,320,320]

    # 仅去噪模式
    m2 = LogWaveletDenoise(c1=3, c2=3, threshold_factor=0.3, level=1, downsample=False).to(device)
    y2 = m2(x)
    print(f"Denoise only: {x.shape} → {y2.shape}")     # [2,3,640,640] → [2,3,640,640]