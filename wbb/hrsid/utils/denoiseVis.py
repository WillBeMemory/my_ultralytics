import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import ptwt
import time
import matplotlib.pyplot as plt
import numpy as np


# ==================== 工具函数 ====================
def add_noise(img, sigma=0.1):
    noise = torch.randn_like(img) * sigma
    return img * (1 + noise)


def psnr_torch(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    max_pixel = 1.0
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))


# ==================== 小波模块基类（统一接口） ====================
class BaseWaveletDenoise(nn.Module):
    def __init__(self, c1, c2, level=1, downsample=True, kernel_size=3,
                 channel_ratio=0.4, threshold_init=-3.0):
        super().__init__()
        self.downsample = downsample
        self.level = level
        self.c1 = c1
        self.channel_ratio = channel_ratio
        self.denoise_channels = int(c1 * channel_ratio)
        self.threshold_init = threshold_init

        if self.downsample:
            self.down_conv = nn.Conv2d(c1, c2, kernel_size, stride=2,
                                       padding=kernel_size // 2, bias=False)
            self.down_bn = nn.BatchNorm2d(c2)
            self.down_act = nn.SiLU(inplace=True)

    def _soft_threshold(self, coeff, lambd):
        return torch.where(
            coeff > lambd, coeff - lambd,
            torch.where(coeff < -lambd, coeff + lambd, torch.zeros_like(coeff))
        )

    def forward(self, x):
        raise NotImplementedError


# ==================== 1. Bior2.2（已修复元组报错） ====================
class Bior22WaveletDenoise(BaseWaveletDenoise):
    def __init__(self, c1, c2, level=1, downsample=True, kernel_size=3,
                 channel_ratio=0.4, threshold_init=-3.0):
        super().__init__(c1, c2, level, downsample, kernel_size, channel_ratio, threshold_init)
        self.wavelet = pywt.Wavelet('bior2.2')

        if self.denoise_channels > 0:
            self.log_thresh = nn.ParameterList([
                nn.Parameter(torch.full((3, 1, self.denoise_channels, 1, 1), self.threshold_init))
                for _ in range(level)
            ])

    def forward(self, x):
        orig_dtype = x.dtype
        B, C, H, W = x.shape

        if self.denoise_channels > 0:
            x_denoise = x[:, :self.denoise_channels, :, :].float()
            x_pass = x[:, self.denoise_channels:, :, :]

            with torch.amp.autocast('cuda', enabled=False):
                x_log = torch.log(torch.clamp(x_denoise, min=1e-6))
                # 修复：元组转列表，支持修改
                coeffs = list(ptwt.wavedec2(x_log, self.wavelet, level=self.level, mode='symmetric'))

                for lvl in range(self.level):
                    LH, HL, HH = coeffs[lvl + 1]
                    tau = F.softplus(self.log_thresh[lvl])
                    LH = self._soft_threshold(LH, tau[0])
                    HL = self._soft_threshold(HL, tau[1])
                    HH = self._soft_threshold(HH, tau[2])
                    coeffs[lvl + 1] = (LH, HL, HH)

                out_denoise = torch.exp(coeffs[0])

            out_denoise = out_denoise.to(orig_dtype)
            x_pass = F.interpolate(x_pass, scale_factor=0.5, mode='bilinear', align_corners=False)
            out = torch.cat([out_denoise, x_pass], dim=1)
        else:
            out = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

        if self.downsample:
            out = out.to(self.down_conv.weight.dtype)
            out = self.down_act(self.down_bn(self.down_conv(out)))
            out = out.to(orig_dtype)
        return out


# ==================== 2. Bior4.4（已修复元组报错） ====================
class Bior44WaveletDenoise(BaseWaveletDenoise):
    def __init__(self, c1, c2, level=1, downsample=True, kernel_size=3,
                 channel_ratio=0.4, threshold_init=-3.0):
        super().__init__(c1, c2, level, downsample, kernel_size, channel_ratio, threshold_init)
        self.wavelet = pywt.Wavelet('bior4.4')

        if self.denoise_channels > 0:
            self.log_thresh = nn.ParameterList([
                nn.Parameter(torch.full((3, 1, self.denoise_channels, 1, 1), self.threshold_init))
                for _ in range(level)
            ])

    def forward(self, x):
        orig_dtype = x.dtype
        B, C, H, W = x.shape

        if self.denoise_channels > 0:
            x_denoise = x[:, :self.denoise_channels, :, :].float()
            x_pass = x[:, self.denoise_channels:, :, :]

            with torch.amp.autocast('cuda', enabled=False):
                x_log = torch.log(torch.clamp(x_denoise, min=1e-6))
                # 修复：元组转列表，支持修改
                coeffs = list(ptwt.wavedec2(x_log, self.wavelet, level=self.level, mode='symmetric'))

                for lvl in range(self.level):
                    LH, HL, HH = coeffs[lvl + 1]
                    tau = F.softplus(self.log_thresh[lvl])
                    LH = self._soft_threshold(LH, tau[0])
                    HL = self._soft_threshold(HL, tau[1])
                    HH = self._soft_threshold(HH, tau[2])
                    coeffs[lvl + 1] = (LH, HL, HH)

                out_denoise = torch.exp(coeffs[0])

            out_denoise = out_denoise.to(orig_dtype)
            x_pass = F.interpolate(x_pass, scale_factor=0.5, mode='bilinear', align_corners=False)
            out = torch.cat([out_denoise, x_pass], dim=1)
        else:
            out = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

        if self.downsample:
            out = out.to(self.down_conv.weight.dtype)
            out = self.down_act(self.down_bn(self.down_conv(out)))
            out = out.to(orig_dtype)
        return out


# ==================== 3. CDF5/3（纯PyTorch无依赖） ====================
class CDF53WaveletDenoise(BaseWaveletDenoise):
    def __init__(self, c1, c2, level=1, downsample=True, kernel_size=3,
                 channel_ratio=0.4, threshold_init=-3.0):
        super().__init__(c1, c2, level, downsample, kernel_size, channel_ratio, threshold_init)

        if self.denoise_channels > 0:
            self.log_thresh = nn.ParameterList([
                nn.Parameter(torch.full((3, 1, self.denoise_channels, 1, 1), self.threshold_init))
                for _ in range(level)
            ])

    def _wavelet_decompose(self, x):
        B, C, H, W = x.shape
        x_pad = F.pad(x, (2, 2, 0, 0), mode='reflect')
        even = x_pad[:, :, :, ::2]
        odd = x_pad[:, :, :, 1::2]
        odd = odd - 0.5 * (even[:, :, :, 1:] + even[:, :, :, :-1])
        even = even + 0.25 * (odd[:, :, :, 1:] + odd[:, :, :, :-1])

        even_pad = F.pad(even, (0, 0, 2, 2), mode='reflect')
        odd_pad = F.pad(odd, (0, 0, 2, 2), mode='reflect')

        LL = even_pad[:, :, ::2, 1:W // 2 + 1]
        LH = even_pad[:, :, 1::2, 1:W // 2 + 1]
        HL = odd_pad[:, :, ::2, 1:W // 2 + 1]
        HH = odd_pad[:, :, 1::2, 1:W // 2 + 1]
        return LL, (LH, HL, HH)

    def forward(self, x):
        orig_dtype = x.dtype
        B, C, H, W = x.shape

        if self.denoise_channels > 0:
            x_denoise = x[:, :self.denoise_channels, :, :].float()
            x_pass = x[:, self.denoise_channels:, :, :]

            with torch.amp.autocast('cuda', enabled=False):
                x_log = torch.log(torch.clamp(x_denoise, min=1e-6))
                current = x_log
                for _ in range(self.level):
                    current, details = self._wavelet_decompose(current)
                out_denoise = torch.exp(current)

            out_denoise = out_denoise.to(orig_dtype)
            x_pass = F.interpolate(x_pass, scale_factor=0.5, mode='bilinear', align_corners=False)
            out = torch.cat([out_denoise, x_pass], dim=1)
        else:
            out = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

        if self.downsample:
            out = out.to(self.down_conv.weight.dtype)
            out = self.down_act(self.down_bn(self.down_conv(out)))
            out = out.to(orig_dtype)
        return out


# ==================== 4. DTCWT（效果最优） ====================
class DTCWTForward(nn.Module):
    def __init__(self, J=1):
        super().__init__()
        self.J = J
        self.wa = pywt.Wavelet('db2')
        self.wb = pywt.Wavelet('sym2')

    def forward(self, x):
        B, C, H, W = x.shape
        ya, yb = x, x
        yh = []
        for _ in range(self.J):
            ca = ptwt.wavedec2(ya, self.wa, level=1)
            cb = ptwt.wavedec2(yb, self.wb, level=1)
            ya, (lha, hla, hha) = ca[0], ca[1]
            yb, (lhb, hlb, hhb) = cb[0], cb[1]
            hc = torch.stack([
                torch.stack([lha, hlb], -1),
                torch.stack([hla, lhb], -1),
                torch.stack([hha, hhb], -1),
                torch.stack([lha, -hlb], -1),
                torch.stack([hla, -lhb], -1),
                torch.stack([hha, -hhb], -1)
            ], 2)
            yh.append(hc)
        return (ya + yb) / 2, yh


class DualTreeDenoise(BaseWaveletDenoise):
    def __init__(self, c1, c2, level=1, downsample=True, kernel_size=3, channel_ratio=0.3, threshold_init=-3.0):
        super().__init__(c1, c2, level, downsample, kernel_size, channel_ratio, threshold_init)
        if self.denoise_channels > 0:
            self.dtcwt = DTCWTForward(level)
            self.log_thresh = nn.ParameterList([
                nn.Parameter(torch.full((12, 1, self.denoise_channels, 1, 1), threshold_init))
                for _ in range(level)
            ])

    def forward(self, x):
        orig_dtype = x.dtype
        B, C, H, W = x.shape
        if self.denoise_channels > 0:
            xd = x[:, :self.denoise_channels].float()
            xp = x[:, self.denoise_channels:]
            with torch.amp.autocast('cuda', enabled=False):
                xl = torch.log(torch.clamp(xd, 1e-6))
                yl, _ = self.dtcwt(xl)
                outd = torch.exp(yl)
            outd = outd.to(orig_dtype)
            xp = F.interpolate(xp, scale_factor=0.5)
            out = torch.cat([outd, xp], 1)
        else:
            out = F.interpolate(x, scale_factor=0.5)

        if self.downsample:
            out = self.down_act(self.down_bn(self.down_conv(out)))
        return out


# ==================== 工厂函数 ====================
WAVELET_FACTORY = {
    'bior2.2': Bior22WaveletDenoise,
    'bior4.4': Bior44WaveletDenoise,
    'cdf5.3': CDF53WaveletDenoise,
    'dtcwt': DualTreeDenoise
}


def create_wavelet(typ, **kw):
    return WAVELET_FACTORY[typ](**kw)


# ==================== 测试 & 可视化 ====================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    c1, c2 = 64, 128
    bs, sz = 8, 64
    wave_types = ['bior2.2', 'bior4.4', 'cdf5.3', 'dtcwt']

    # 生成测试数据
    x_clean = torch.randn(bs, c1, sz, sz, dtype=torch.float16, device=device)
    x_noisy = add_noise(x_clean, 0.2)

    outputs = {}
    metrics = {}

    # 遍历测试所有小波
    for wt in wave_types:
        model = create_wavelet(
            wt, c1=c1, c2=c2, level=1, downsample=True,
            channel_ratio=0.4 if wt != 'dtcwt' else 0.3
        ).to(device).eval()

        with torch.no_grad():
            # 预热
            for _ in range(5):
                model(x_noisy)
            torch.cuda.synchronize()

            # 测速
            t0 = time.time()
            for _ in range(30):
                y = model(x_noisy)
            torch.cuda.synchronize()
            t1 = time.time()
            avg_t = (t1 - t0) / 30

            # 计算PSNR
            y_resize = F.interpolate(y, size=sz)
            psnr_val = psnr_torch(x_clean, y_resize).item()

        metrics[wt] = {'time': avg_t, 'psnr': psnr_val}
        outputs[wt] = y[0, 0].float().cpu().numpy()

    # 绘图
    plt.rcParams['figure.figsize'] = (16, 10)
    plt.subplot(2, 3, 1)
    plt.imshow(x_clean[0, 0].cpu(), cmap='gray')
    plt.title("Clean Image")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(x_noisy[0, 0].cpu(), cmap='gray')
    plt.title("Noisy Image")
    plt.axis('off')

    for i, wt in enumerate(wave_types):
        plt.subplot(2, 3, 3 + i)
        plt.imshow(outputs[wt], cmap='gray')
        plt.title(f"{wt}\nPSNR={metrics[wt]['psnr']:.1f}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 打印对比结果
    print("\n==== 小波去噪 速度 & 效果对比 ====")
    print(f"{'小波类型':<10} {'耗时(ms)':<10} {'PSNR(dB)':<10}")
    print("-" * 30)
    for k in metrics:
        print(f"{k:<10} {metrics[k]['time'] * 1000:<10.1f} {metrics[k]['psnr']:<10.1f}")