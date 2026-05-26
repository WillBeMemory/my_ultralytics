import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pywt
import ptwt

class LogWaveletDenoise(nn.Module):
    def __init__(self, c1, c2, level=1, downsample=True, kernel_size=3):
        super().__init__()
        self.downsample = downsample
        self.level = level
        self.wavelet_a = 'db2'
        self.wavelet_b = 'sym2'

        self.log_thresh = nn.ParameterList()
        for _ in range(level):
            for _ in range(6):
                self.log_thresh.append(nn.Parameter(torch.full((1, c1, 1, 1), -2.0)))

        if self.downsample:
            self.down_conv = nn.Conv2d(c1, c2, kernel_size, stride=2,
                                       padding=kernel_size // 2, bias=False)
            self.down_bn = nn.BatchNorm2d(c2)
            self.down_act = nn.SiLU(inplace=True)

    def _soft_threshold(self, coeff, lambd):
        return torch.where(
            coeff > lambd,
            coeff - lambd,
            torch.where(coeff < -lambd, coeff + lambd, torch.zeros_like(coeff))
        )

    def _multilevel_denoise(self, x_log):
        B, C, H, W = x_log.shape
        details_a = []
        details_b = []
        current_a = x_log
        current_b = x_log

        for lvl in range(self.level):
            # 在 wavedec2 中保留 mode 参数
            coeffs_a = ptwt.wavedec2(current_a, pywt.Wavelet(self.wavelet_a), level=1, mode='symmetric')
            coeffs_b = ptwt.wavedec2(current_b, pywt.Wavelet(self.wavelet_b), level=1, mode='symmetric')
            approx_a, detail_a = coeffs_a[0], coeffs_a[1]
            approx_b, detail_b = coeffs_b[0], coeffs_b[1]

            details_a.append((detail_a, current_a.shape[2:]))
            details_b.append((detail_b, current_b.shape[2:]))
            current_a = approx_a
            current_b = approx_b

        recon_a = current_a
        recon_b = current_b

        for lvl in range(self.level - 1, -1, -1):
            high_a, orig_size = details_a[lvl]
            high_b, _ = details_b[lvl]

            base = lvl * 6
            tau_a_LH = F.softplus(self.log_thresh[base])
            tau_a_HL = F.softplus(self.log_thresh[base + 1])
            tau_a_HH = F.softplus(self.log_thresh[base + 2])
            tau_b_LH = F.softplus(self.log_thresh[base + 3])
            tau_b_HL = F.softplus(self.log_thresh[base + 4])
            tau_b_HH = F.softplus(self.log_thresh[base + 5])

            LH_a, HL_a, HH_a = high_a
            LH_a = self._soft_threshold(LH_a, tau_a_LH)
            HL_a = self._soft_threshold(HL_a, tau_a_HL)
            HH_a = self._soft_threshold(HH_a, tau_a_HH)
            high_a_thresh = (LH_a, HL_a, HH_a)

            LH_b, HL_b, HH_b = high_b
            LH_b = self._soft_threshold(LH_b, tau_b_LH)
            HL_b = self._soft_threshold(HL_b, tau_b_HL)
            HH_b = self._soft_threshold(HH_b, tau_b_HH)
            high_b_thresh = (LH_b, HL_b, HH_b)

            # 移除了 waverec2 的 mode 参数
            recon_a = ptwt.waverec2([recon_a, high_a_thresh], pywt.Wavelet(self.wavelet_a))
            recon_b = ptwt.waverec2([recon_b, high_b_thresh], pywt.Wavelet(self.wavelet_b))

            if recon_a.shape[2] != orig_size[0] or recon_a.shape[3] != orig_size[1]:
                recon_a = recon_a[:, :, :orig_size[0], :orig_size[1]]
                recon_b = recon_b[:, :, :orig_size[0], :orig_size[1]]

        recon = (recon_a + recon_b) / 2.0
        if recon.shape[2] != H or recon.shape[3] != W:
            recon = recon[:, :, :H, :W]
        return recon

    def forward(self, x):
        x_log = torch.log(torch.clamp(x, min=1e-6))
        rec_log = self._multilevel_denoise(x_log)
        out = torch.exp(rec_log)

        if self.downsample:
            out = self.down_act(self.down_bn(self.down_conv(out)))
        return out


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = LogWaveletDenoise(c1=64, c2=128, level=3, downsample=True, kernel_size=3).to(device)
    model.train()

    x = torch.randn(2, 64, 32, 32).to(device)
    y = model(x)
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {y.shape}")

    loss = y.mean()
    loss.backward()
    print("Backward passed. Test OK!")