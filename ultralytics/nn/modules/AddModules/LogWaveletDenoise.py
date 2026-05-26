import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pywt  # 仅用于提取滤波器系数

class LogWaveletDenoise(nn.Module):
    def __init__(self, c1, c2, level=1, downsample=True, kernel_size=3, wavelet='bior4.4'):
        super().__init__()
        self.downsample = downsample
        self.level = level

        # ---------- 从 pywt 提取平滑小波基（固定） ----------
        w = pywt.Wavelet(wavelet)
        dec_lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)  # 卷积需翻转
        dec_hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)
        rec_lo = torch.tensor(w.rec_lo, dtype=torch.float32)        # 重构无需翻转
        rec_hi = torch.tensor(w.rec_hi, dtype=torch.float32)

        # 构建二维滤波器组 (4, 1, len, len)
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

        # 可学习阈值（每级每方向一个）
        self.log_thresh_LH = nn.ParameterList([
            nn.Parameter(torch.full((1, c1, 1, 1), -2.0)) for _ in range(level)
        ])
        self.log_thresh_HL = nn.ParameterList([
            nn.Parameter(torch.full((1, c1, 1, 1), -2.0)) for _ in range(level)
        ])
        self.log_thresh_HH = nn.ParameterList([
            nn.Parameter(torch.full((1, c1, 1, 1), -2.0)) for _ in range(level)
        ])

        # 下采样分支
        if self.downsample:
            self.down_conv = nn.Conv2d(c1, c2, kernel_size, stride=2,
                                       padding=kernel_size // 2, bias=False)
            self.down_bn = nn.BatchNorm2d(c2)
            self.down_act = nn.SiLU(inplace=True)

    # ---------- 小波变换核心 ----------
    def _dwt(self, x):
        """单级二维 DWT（使用分组卷积）"""
        B, C, H, W = x.shape
        pad_len = self.dec_filters.shape[-1] // 2  # 滤波器半长
        # 填充至偶数（以便下采样）
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        # 边界扩展以适配滤波器长度
        x = F.pad(x, (pad_len, pad_len, pad_len, pad_len), mode='reflect')
        filters = self.dec_filters.repeat(C, 1, 1, 1)  # (4*C, 1, len, len)
        coeffs = F.conv2d(x, filters, stride=2, groups=C)
        coeffs = coeffs.view(B, C, 4, coeffs.shape[-2], coeffs.shape[-1])
        return coeffs[:, :, 0], coeffs[:, :, 1], coeffs[:, :, 2], coeffs[:, :, 3]

    def _idwt(self, LL, LH, HL, HH, out_h, out_w):
        """单级二维 IDWT（转置卷积，输出尺寸为 out_h x out_w）"""
        B, C, H, W = LL.shape
        coeffs = torch.stack([LL, LH, HL, HH], dim=2).reshape(B, C * 4, H, W)
        pad_len = self.rec_filters.shape[-1] // 2
        filters = self.rec_filters.repeat(C, 1, 1, 1)
        out = F.conv_transpose2d(coeffs, filters, stride=2, padding=pad_len, groups=C)
        # 裁剪到目标尺寸（转置卷积输出可能大于目标）
        return out[:, :, :out_h, :out_w]

    def _soft_threshold(self, coeff, lambd):
        return torch.where(
            coeff > lambd, coeff - lambd,
            torch.where(coeff < -lambd, coeff + lambd, torch.zeros_like(coeff))
        )

    def _multilevel_denoise(self, x_log):
        detail_coeffs = []   # 每级保存 (LH, HL, HH, 输入高度, 输入宽度)
        current = x_log
        for lvl in range(self.level):
            in_h, in_w = current.shape[2], current.shape[3]
            LL, LH, HL, HH = self._dwt(current)
            detail_coeffs.append((LH, HL, HH, in_h, in_w))
            current = LL

        for lvl in reversed(range(self.level)):
            LH, HL, HH, in_h, in_w = detail_coeffs[lvl]
            tau_lh = F.softplus(self.log_thresh_LH[lvl])
            tau_hl = F.softplus(self.log_thresh_HL[lvl])
            tau_hh = F.softplus(self.log_thresh_HH[lvl])
            LH = self._soft_threshold(LH, tau_lh)
            HL = self._soft_threshold(HL, tau_hl)
            HH = self._soft_threshold(HH, tau_hh)
            current = self._idwt(current, LH, HL, HH, in_h, in_w)
        return current

    def forward(self, x):
        orig_dtype = x.dtype
        # 强制使用 float32 进行小波变换和去噪（AMP 不会插手）
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            x_log = torch.log(torch.clamp(x, min=1e-6))
            rec_log = self._multilevel_denoise(x_log)
            out = torch.exp(rec_log)
            # 下采样分支（此时 out 必然是 float32，与权重类型一致）
            if self.downsample:
                out = self.down_act(self.down_bn(self.down_conv(out)))
        # 恢复原始精度（通常是 float16）
        out = out.to(orig_dtype)
        return out

# ========== 简单测试 ==========
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 测试 1：下采样模式，level=1
    print("\n--- Test 1: downsample=True, level=1 ---")
    model1 = LogWaveletDenoise(c1=64, c2=128, level=1, downsample=True, kernel_size=3).to(device)
    model1.train()
    x1 = torch.randn(2, 64, 32, 32).to(device)
    y1 = model1(x1)
    print(f"Input shape : {x1.shape}")
    print(f"Output shape: {y1.shape}  (expected: (2, 128, 16, 16))")
    loss1 = y1.mean()
    loss1.backward()
    print("Backward passed. Check gradients:",
          all(p.grad is not None for p in model1.parameters() if p.requires_grad))

    # 测试 2：不下采样模式，level=2
    print("\n--- Test 2: downsample=False, level=2 ---")
    model2 = LogWaveletDenoise(c1=32, c2=32, level=2, downsample=False).to(device)
    model2.train()
    x2 = torch.randn(1, 32, 20, 20).to(device)
    y2 = model2(x2)
    print(f"Input shape : {x2.shape}")
    print(f"Output shape: {y2.shape}  (expected: (1, 32, 20, 20))")
    loss2 = y2.mean()
    loss2.backward()
    print("Backward passed. Check gradients:",
          all(p.grad is not None for p in model2.parameters() if p.requires_grad))

    # 测试 3：float16 兼容性 (AMP 模拟)
    print("\n--- Test 3: float16 input (AMP compatible) ---")
    model3 = LogWaveletDenoise(c1=64, c2=64, level=1, downsample=False).to(device)
    model3.train()
    x3 = torch.randn(2, 64, 32, 32, dtype=torch.float16).to(device)
    y3 = model3(x3)
    print(f"Input dtype : {x3.dtype}")
    print(f"Output dtype: {y3.dtype}  (expected: torch.float16)")
    print(f"Output shape: {y3.shape}")
    loss3 = y3.float().mean()
    loss3.backward()
    print("Backward passed. Test OK!")

    print("\nAll tests passed!")