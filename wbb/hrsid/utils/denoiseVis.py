import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import math
import pywt
import ptwt   # 需要安装: pip install ptwt

# -------------------- 通用软阈值与噪声估计 --------------------
def soft_threshold_torch(coeff, lambd):
    return torch.where(coeff > lambd, coeff - lambd,
                       torch.where(coeff < -lambd, coeff + lambd, torch.zeros_like(coeff)))

def estimate_noise_sigma_torch(HH):
    hh_abs = torch.abs(HH.reshape(HH.shape[0], -1))
    median = torch.median(hh_abs, dim=1).values
    return median / 0.6745

# -------------------- Haar DWT --------------------
def haar_filters():
    lo = torch.tensor([1.0, 1.0]) / math.sqrt(2)
    hi = torch.tensor([1.0, -1.0]) / math.sqrt(2)
    LL = (lo.unsqueeze(0) * lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    LH = (lo.unsqueeze(0) * hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    HL = (hi.unsqueeze(0) * lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    HH = (hi.unsqueeze(0) * hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    return torch.cat([LL, LH, HL, HH], dim=0)

def dwt_haar(x, dec_filters):
    dec_filters = dec_filters.to(x.device)
    B, C, H, W = x.shape
    pad_h = (2 - H % 2) % 2
    pad_w = (2 - W % 2) % 2
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    filters = dec_filters.repeat(C, 1, 1, 1)
    coeffs = F.conv2d(x, filters, stride=2, groups=C)
    coeffs = coeffs.view(B, C, 4, coeffs.shape[-2], coeffs.shape[-1])
    return coeffs[:, :, 0], coeffs[:, :, 1], coeffs[:, :, 2], coeffs[:, :, 3]

def idwt_haar(LL, LH, HL, HH, rec_filters):
    rec_filters = rec_filters.to(LL.device)
    B, C, H, W = LL.shape
    coeffs = torch.stack([LL, LH, HL, HH], dim=2).reshape(B, C * 4, H, W)
    filters = rec_filters.repeat(C, 1, 1, 1)
    out = F.conv_transpose2d(coeffs, filters, stride=2, groups=C)
    return out[:, :, :2*H, :2*W]

def denoise_haar(img_tensor, level=1, threshold_factor=0.5):
    device = img_tensor.device
    dtype = img_tensor.dtype
    dec = haar_filters().to(device, dtype)
    rec = dec.clone()
    x_log = torch.log(torch.clamp(img_tensor, min=1e-6))
    current = x_log
    coeffs_list = []
    for _ in range(level):
        LL, LH, HL, HH = dwt_haar(current, dec)
        coeffs_list.append((LH, HL, HH))
        current = LL
    for lvl in range(level-1, -1, -1):
        LH, HL, HH = coeffs_list[lvl]
        sigma = estimate_noise_sigma_torch(HH)
        N = HH[0].numel()
        lambd = sigma * math.sqrt(2.0 * math.log(N)) * threshold_factor
        lambd = lambd.view(-1, 1, 1, 1)
        LH = soft_threshold_torch(LH, lambd)
        HL = soft_threshold_torch(HL, lambd)
        HH = soft_threshold_torch(HH, lambd)
        current = idwt_haar(current, LH, HL, HH, rec)
    rec_log = current[:, :, :x_log.shape[2], :x_log.shape[3]]
    out = torch.exp(rec_log)
    return out

# -------------------- Bior4.4 DWT --------------------
def bior44_filters():
    w = pywt.Wavelet('bior4.4')
    lo_d, hi_d = torch.tensor(w.dec_lo, dtype=torch.float32), torch.tensor(w.dec_hi, dtype=torch.float32)
    lo_r, hi_r = torch.tensor(w.rec_lo, dtype=torch.float32), torch.tensor(w.rec_hi, dtype=torch.float32)
    LL_d = (lo_d.unsqueeze(0) * lo_d.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    LH_d = (lo_d.unsqueeze(0) * hi_d.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    HL_d = (hi_d.unsqueeze(0) * lo_d.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    HH_d = (hi_d.unsqueeze(0) * hi_d.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    dec = torch.cat([LL_d, LH_d, HL_d, HH_d], dim=0)
    LL_r = (lo_r.unsqueeze(0) * lo_r.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    LH_r = (lo_r.unsqueeze(0) * hi_r.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    HL_r = (hi_r.unsqueeze(0) * lo_r.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    HH_r = (hi_r.unsqueeze(0) * hi_r.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    rec = torch.cat([LL_r, LH_r, HL_r, HH_r], dim=0)
    return dec, rec

def dwt_bior44(x, dec_filters):
    dec_filters = dec_filters.to(x.device)
    B, C, H, W = x.shape
    orig_h, orig_w = H, W
    pad_h = (2 - H % 2) % 2
    pad_w = (2 - W % 2) % 2
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    filter_pad = dec_filters.shape[-1] // 2
    x = F.pad(x, (filter_pad, filter_pad, filter_pad, filter_pad), mode='reflect')
    filters = dec_filters.repeat(C, 1, 1, 1)
    coeffs = F.conv2d(x, filters, stride=2, groups=C)
    coeffs = coeffs.view(B, C, 4, coeffs.shape[-2], coeffs.shape[-1])
    return (coeffs[:, :, 0], coeffs[:, :, 1], coeffs[:, :, 2], coeffs[:, :, 3]), orig_h, orig_w

def idwt_bior44(LL, LH, HL, HH, rec_filters, orig_h, orig_w):
    rec_filters = rec_filters.to(LL.device)
    B, C, H, W = LL.shape
    coeffs = torch.stack([LL, LH, HL, HH], dim=2).reshape(B, C * 4, H, W)
    filter_pad = rec_filters.shape[-1] // 2
    filters = rec_filters.repeat(C, 1, 1, 1)
    out = F.conv_transpose2d(coeffs, filters, stride=2, padding=filter_pad, groups=C)
    return out[:, :, :orig_h, :orig_w]

def denoise_bior44(img_tensor, level=1, threshold_factor=0.5):
    device = img_tensor.device
    dtype = img_tensor.dtype
    dec, rec = bior44_filters()
    dec, rec = dec.to(device, dtype), rec.to(device, dtype)
    x_log = torch.log(torch.clamp(img_tensor, min=1e-6))
    orig_h, orig_w = x_log.shape[2], x_log.shape[3]
    current = x_log
    coeffs_list = []
    for _ in range(level):
        (LL, LH, HL, HH), cur_h, cur_w = dwt_bior44(current, dec)
        coeffs_list.append((LH, HL, HH, cur_h, cur_w))
        current = LL
    for lvl in range(level-1, -1, -1):
        LH, HL, HH, cur_h, cur_w = coeffs_list[lvl]
        sigma = estimate_noise_sigma_torch(HH)
        N = HH[0].numel()
        lambd = sigma * math.sqrt(2.0 * math.log(N)) * threshold_factor
        lambd = lambd.view(-1, 1, 1, 1)
        LH = soft_threshold_torch(LH, lambd)
        HL = soft_threshold_torch(HL, lambd)
        HH = soft_threshold_torch(HH, lambd)
        current = idwt_bior44(current, LH, HL, HH, rec, cur_h, cur_w)
    rec_log = current[:, :, :orig_h, :orig_w]
    out = torch.exp(rec_log)
    return out

# -------------------- DTCWT 近似 (基于pywt的多级DWT, 在pixel域和log域) --------------------
def denoise_dtcwt_pixel(img_np, level=3, threshold_factor=0.5):
    img_255 = np.clip(img_np * 255.0, 0, 255).astype(np.float32)
    out = np.zeros_like(img_255)
    for c in range(3):
        coeffs = pywt.wavedec2(img_255[:,:,c], 'bior4.4', level=level)
        sigma = (np.median(np.abs(coeffs[-1][-1])) / 0.6745)
        N = sum(band.size for coeff in coeffs[1:] for band in coeff)
        lambd = sigma * math.sqrt(2.0 * math.log(N)) * threshold_factor
        def sth(x): return np.sign(x) * np.maximum(np.abs(x) - lambd, 0)
        coeffs[1:] = [tuple(sth(band) for band in level_coeff) for level_coeff in coeffs[1:]]
        rec = pywt.waverec2(coeffs, 'bior4.4')
        out[:,:,c] = rec
    return np.clip(out / 255.0, 0, 1)

def denoise_dtcwt_log(img_np, level=3, threshold_factor=0.5):
    img_log = np.log(np.clip(img_np, 1e-6, None))
    log_min, log_max = img_log.min(), img_log.max()
    img_mapped = (img_log - log_min) / (log_max - log_min + 1e-8) * 255.0
    denoised_mapped = denoise_dtcwt_pixel(img_mapped / 255.0, level=level, threshold_factor=threshold_factor) * 255.0
    denoised_log = denoised_mapped / 255.0 * (log_max - log_min) + log_min
    return np.exp(denoised_log)

# -------------------- 基于ptwt的双树小波变换（复数域，可微分） --------------------
class DTCWTWithPTWT:
    def __init__(self, wavelet_a='db2', wavelet_b='db2', level=3):
        self.wavelet_a = wavelet_a
        self.wavelet_b = wavelet_b
        self.level = level

    def forward(self, data):
        coeffs_a = ptwt.wavedec2(data, self.wavelet_a, level=self.level)
        coeffs_b = ptwt.wavedec2(data, self.wavelet_b, level=self.level)
        lowpass = coeffs_a[0] + 1j * coeffs_b[0]
        highpasses = []
        for level_idx in range(1, len(coeffs_a)):
            band_a = coeffs_a[level_idx]
            band_b = coeffs_b[level_idx]
            complex_band = (band_a[0] + 1j * band_b[0],
                            band_a[1] + 1j * band_b[1],
                            band_a[2] + 1j * band_b[2])
            highpasses.append(complex_band)
        return lowpass, highpasses

    def inverse(self, lowpass, highpasses):
        real_low = torch.real(lowpass)
        imag_low = torch.imag(lowpass)
        real_high_list = []
        imag_high_list = []
        for complex_band in highpasses:
            real_high = (torch.real(complex_band[0]),
                         torch.real(complex_band[1]),
                         torch.real(complex_band[2]))
            imag_high = (torch.imag(complex_band[0]),
                         torch.imag(complex_band[1]),
                         torch.imag(complex_band[2]))
            real_high_list.append(real_high)
            imag_high_list.append(imag_high)
        coeffs_a = [real_low] + real_high_list
        coeffs_b = [imag_low] + imag_high_list
        recon_a = ptwt.waverec2(coeffs_a, self.wavelet_a)
        recon_b = ptwt.waverec2(coeffs_b, self.wavelet_b)
        return (recon_a + recon_b) / 2.0

def denoise_dtcwt_ptwt_log(img_tensor, level=3, threshold_factor=0.5):
    """对数域双树小波去噪（基于ptwt）"""
    device = img_tensor.device
    x_log = torch.log(torch.clamp(img_tensor, min=1e-6))
    dtcwt = DTCWTWithPTWT(wavelet_a='db2', wavelet_b='db2', level=level)
    lowpass, highpasses = dtcwt.forward(x_log)

    # 噪声估计（使用最高频对角子带的模值中位数）
    diag_high = highpasses[-1][2]
    diag_abs = torch.abs(diag_high)
    median_abs = torch.median(diag_abs.reshape(diag_abs.shape[0], -1), dim=1).values
    sigma = median_abs / 0.6745
    N = diag_high[0].numel()
    lambd = sigma * math.sqrt(2.0 * math.log(N)) * threshold_factor
    lambd = lambd.view(-1, 1, 1, 1)

    def soft_threshold_complex(coeff_complex, lambd):
        real = torch.real(coeff_complex)
        imag = torch.imag(coeff_complex)
        real_thresh = soft_threshold_torch(real, lambd)
        imag_thresh = soft_threshold_torch(imag, lambd)
        return real_thresh + 1j * imag_thresh

    lowpass = soft_threshold_complex(lowpass, lambd)
    highpasses_thresh = []
    for band in highpasses:
        thresh_band = (soft_threshold_complex(band[0], lambd),
                       soft_threshold_complex(band[1], lambd),
                       soft_threshold_complex(band[2], lambd))
        highpasses_thresh.append(thresh_band)

    recon_log = dtcwt.inverse(lowpass, highpasses_thresh)
    recon_log = recon_log[:, :, :x_log.shape[2], :x_log.shape[3]]
    out = torch.exp(recon_log)
    out = torch.clamp(out, 0, 1)
    return out

def denoise_dtcwt_ptwt_pixel(img_tensor, level=3, threshold_factor=0.5):
    """像素域双树小波去噪（基于ptwt）"""
    device = img_tensor.device
    dtcwt = DTCWTWithPTWT(wavelet_a='db2', wavelet_b='db2', level=level)
    lowpass, highpasses = dtcwt.forward(img_tensor)

    diag_high = highpasses[-1][2]
    diag_abs = torch.abs(diag_high)
    median_abs = torch.median(diag_abs.reshape(diag_abs.shape[0], -1), dim=1).values
    sigma = median_abs / 0.6745
    N = diag_high[0].numel()
    lambd = sigma * math.sqrt(2.0 * math.log(N)) * threshold_factor
    lambd = lambd.view(-1, 1, 1, 1)

    def soft_threshold_complex(coeff_complex, lambd):
        real = torch.real(coeff_complex)
        imag = torch.imag(coeff_complex)
        real_thresh = soft_threshold_torch(real, lambd)
        imag_thresh = soft_threshold_torch(imag, lambd)
        return real_thresh + 1j * imag_thresh

    lowpass = soft_threshold_complex(lowpass, lambd)
    highpasses_thresh = []
    for band in highpasses:
        thresh_band = (soft_threshold_complex(band[0], lambd),
                       soft_threshold_complex(band[1], lambd),
                       soft_threshold_complex(band[2], lambd))
        highpasses_thresh.append(thresh_band)

    recon = dtcwt.inverse(lowpass, highpasses_thresh)
    recon = recon[:, :, :img_tensor.shape[2], :img_tensor.shape[3]]
    recon = torch.clamp(recon, 0, 1)
    return recon

# -------------------- 标签与目标框 --------------------
def load_yolo_labels(label_path):
    targets = []
    if not os.path.exists(label_path):
        return targets
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = map(float, parts)
            targets.append((cx, cy, w, h))
    return targets

def draw_gt_boxes(ax, gt_boxes, img_size):
    for cx, cy, w, h in gt_boxes:
        x1 = (cx - w/2) * img_size
        y1 = (cy - h/2) * img_size
        width = w * img_size
        height = h * img_size
        rect = patches.Rectangle((x1, y1), width, height, linewidth=1.5,
                                 edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

# -------------------- 主对比函数（7列） --------------------
def compare_all_methods(
    input_dir=r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_YOLO\images\train",
    labels_dir=r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_YOLO\labels\train",
    output_dir=r"./denoised_comparison_all",
    num_images=5,
    img_size=640,
    decomp_level=2,
    dtcwt_level=3,
    threshold_factor=0.5,
):
    os.makedirs(output_dir, exist_ok=True)
    all_imgs = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.bmp'))]
    selected = sorted(all_imgs)[:num_images]
    print(f"Comparing methods (including pixel-domain ptwt DTCWT) on {len(selected)} images...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for idx, img_name in enumerate(selected):
        img_path = os.path.join(input_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((img_size, img_size), Image.BILINEAR)
        img_np = np.array(img).astype(np.float64) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')
        gt_boxes = load_yolo_labels(label_path)

        with torch.no_grad():
            haar_np = denoise_haar(img_tensor, level=decomp_level, threshold_factor=threshold_factor).squeeze(0).permute(1,2,0).cpu().numpy()
            bior_np = denoise_bior44(img_tensor, level=decomp_level, threshold_factor=threshold_factor).squeeze(0).permute(1,2,0).cpu().numpy()
            dtcwt_ptwt_log_np = denoise_dtcwt_ptwt_log(img_tensor, level=dtcwt_level, threshold_factor=threshold_factor).squeeze(0).permute(1,2,0).cpu().numpy()
            dtcwt_ptwt_pixel_np = denoise_dtcwt_ptwt_pixel(img_tensor, level=dtcwt_level, threshold_factor=threshold_factor).squeeze(0).permute(1,2,0).cpu().numpy()

        dtcwt_log_np = denoise_dtcwt_log(img_np, level=dtcwt_level, threshold_factor=threshold_factor)
        dtcwt_pixel_np = denoise_dtcwt_pixel(img_np, level=dtcwt_level, threshold_factor=threshold_factor)

        # 7列对比
        fig, axes = plt.subplots(1, 7, figsize=(42, 7))
        titles = ['Original', 'DTCWT (ptwt, log)', 'DTCWT (ptwt, pixel)',
                  'DTCWT (log)', 'DTCWT (pixel)', 'Haar DWT', 'Bior4.4 DWT']
        images = [img_np, dtcwt_ptwt_log_np, dtcwt_ptwt_pixel_np,
                  dtcwt_log_np, dtcwt_pixel_np, haar_np, bior_np]

        for ax, title, im in zip(axes, titles, images):
            ax.imshow(np.clip(im, 0, 1))
            ax.set_title(title, fontsize=10)
            ax.axis('off')

        if gt_boxes:
            draw_gt_boxes(axes[0], gt_boxes, img_size)

        plt.tight_layout(pad=1.5)
        save_path = os.path.join(output_dir, f"compare_{os.path.splitext(img_name)[0]}.png")
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved {save_path}")

    print("All comparisons completed. Output:", output_dir)

if __name__ == "__main__":
    compare_all_methods()