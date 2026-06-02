import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LogWaveletDenoise(nn.Module):
    """Log-domain wavelet denoising with spatial-adaptive threshold.

    Key design:
    - log(x) converts multiplicative SAR speckle to additive noise
    - Haar DWT decomposes into subbands
    - Per-channel learnable soft-threshold (via softplus) for each detail subband
    - Spatial-adaptive threshold: LL local variance modulates per-position
      strength (textured→lower threshold, smooth→higher threshold)
    - No residual fusion (pure denoised output)
    - Full float32 computation under AMP

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels (after downsample).
        level (int): Wavelet decomposition levels (default 1).
        downsample (bool): Apply stride-2 conv after denoising.
        kernel_size (int): Downsample conv kernel size.
    """

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

        # ---------- 可学习阈值（per-channel, per-subband, per-level） ----------
        # init log_thresh = -2.0 → softplus(-2.0) ≈ 0.127 (nearly no denoising)
        self.log_thresh_LH = nn.ParameterList([
            nn.Parameter(torch.full((1, c1, 1, 1), -2.0)) for _ in range(level)
        ])
        self.log_thresh_HL = nn.ParameterList([
            nn.Parameter(torch.full((1, c1, 1, 1), -2.0)) for _ in range(level)
        ])
        self.log_thresh_HH = nn.ParameterList([
            nn.Parameter(torch.full((1, c1, 1, 1), -2.0)) for _ in range(level)
        ])

        # ---------- 噪声估计基值（首batch初始化，后由学习阈值补偿） ----------
        self.register_buffer('noise_sigma', torch.ones(level, c1) * 0.1)
        self.register_buffer('noise_initialized', torch.tensor(False))

        # ---------- 空间自适应阈值参数 ----------
        self.log_spatial_bias = nn.Parameter(torch.tensor(-2.0))
        self.spatial_sensitivity = nn.Parameter(torch.tensor(0.1))

        # ---------- 可选下采样 ----------
        if self.downsample:
            self.down_conv = nn.Conv2d(c1, c2, kernel_size, stride=2,
                                       padding=kernel_size // 2, bias=False)
            self.down_bn = nn.BatchNorm2d(c2)
            self.down_act = nn.SiLU(inplace=True)

    # --- 小波分解/重构 ---
    def _dwt(self, x):
        B, C, H, W = x.shape
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        filters = self.dec_filters.to(x.dtype).repeat(C, 1, 1, 1)
        coeffs = F.conv2d(x, filters, stride=2, groups=C)
        coeffs = coeffs.view(B, C, 4, coeffs.shape[-2], coeffs.shape[-1])
        return coeffs[:, :, 0], coeffs[:, :, 1], coeffs[:, :, 2], coeffs[:, :, 3]

    def _idwt(self, LL, LH, HL, HH, orig_h, orig_w):
        B, C, H, W = LL.shape
        coeffs = torch.stack([LL, LH, HL, HH], dim=2).reshape(B, C * 4, H, W)
        filters = self.rec_filters.to(LL.dtype).repeat(C, 1, 1, 1)
        out = F.conv_transpose2d(coeffs, filters, stride=2, groups=C)
        return out[:, :, :orig_h, :orig_w]

    def _estimate_noise_base(self, HH):
        """MAD-based noise estimation from HH subband (Donoho threshold)."""
        hh_abs = torch.abs(HH.reshape(HH.size(0), HH.size(1), -1))
        sorted_abs = torch.sort(hh_abs, dim=-1)[0]
        N = hh_abs.shape[-1]
        k = N // 2
        sigma = sorted_abs[:, :, k] / 0.6745
        return sigma * math.sqrt(2.0 * math.log(N + 1e-6))

    def _soft_threshold(self, coeff, lambd):
        return torch.where(
            coeff > lambd, coeff - lambd,
            torch.where(coeff < -lambd, coeff + lambd, torch.zeros_like(coeff))
        )

    def _spatial_scale(self, ll):
        """Compute spatial-adaptive threshold scale from LL local variance.

        Args:
            ll: Low-frequency subband (B, C, H_ll, W_ll).

        Returns:
            spatial_scale: (B, C, H_ll, W_ll), values in (0,1] for textured,
                           values in (1, ~2) for smooth.
        """
        # Local 3×3 variance
        ll_mean = F.avg_pool2d(ll, 3, stride=1, padding=1)
        ll_sq_mean = F.avg_pool2d(ll ** 2, 3, stride=1, padding=1)
        ll_var = (ll_sq_mean - ll_mean ** 2).clamp(min=0)

        # sigmoid 工作点: bias=-2 → var=0时 sigmoid(-2)≈0.12 (低 → 阈值低)
        #   但 spatial_scale 作为乘子，sigmoid 范围 (0,1)，不够
        #   改为: scale = 0.5 + 1.5 * sigmoid(bias - var * sensitivity)
        #   范围 (0.5, 2.0)，高 var → scale < 1，低 var → scale > 1
        raw = torch.sigmoid(self.log_spatial_bias - ll_var * self.spatial_sensitivity)
        spatial_scale = 0.5 + 1.5 * raw  # (0.5, 2.0)
        return spatial_scale

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

            # Base threshold: softplus(log_thresh) × noise_sigma → (1, C, 1, 1)
            noise = self.noise_sigma[lvl].float().view(1, -1, 1, 1)
            tau_lh = F.softplus(self.log_thresh_LH[lvl]) * noise
            tau_hl = F.softplus(self.log_thresh_HL[lvl]) * noise
            tau_hh = F.softplus(self.log_thresh_HH[lvl]) * noise

            # Spatial-adaptive modulation from LL variance
            spatial = self._spatial_scale(current.float())

            # Apply spatial modulation + soft threshold
            LH = self._soft_threshold(LH, tau_lh * spatial)
            HL = self._soft_threshold(HL, tau_hl * spatial)
            HH = self._soft_threshold(HH, tau_hh * spatial)

            current = self._idwt(current, LH, HL, HH, orig_h, orig_w)

        if current.shape[2] != H or current.shape[3] != W:
            current = current[:, :, :H, :W]
        return current

    def forward(self, x):
        orig_dtype = x.dtype
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

            if self.downsample:
                out = out.to(self.down_conv.weight.dtype)
                out = self.down_act(self.down_bn(self.down_conv(out)))

        return out.to(orig_dtype)


# ================== 简单测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 测试不同输入尺寸
    for c_in, c_out, level, hw in [
        (3,  32, 1, (640, 640)),
        (3,  32, 2, (640, 640)),
        (3,  32, 3, (640, 640)),
        (3,  32, 1, (512, 512)),
        (3,  32, 1, (800, 800)),
    ]:
        x = torch.randn(2, c_in, hw[0], hw[1], device=device)
        model = LogWaveletDenoise(c_in, c_out, level=level, downsample=True).to(device)
        model.train()
        y = model(x)
        expected_h = hw[0] // 2 if model.downsample else hw[0]
        expected_w = hw[1] // 2 if model.downsample else hw[1]
        status = "✅" if y.shape == (2, c_out, expected_h, expected_w) else "❌"
        print(f"level={level}, {hw[0]}×{hw[1]}: {x.shape} → {y.shape} {status}")
        loss = y.mean()
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is None:
                print(f"  ⚠️ No grad for {name}")
    print("Gradients OK")
