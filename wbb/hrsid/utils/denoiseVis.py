import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import math
import pywt
try:
    import scipy.ndimage
except ImportError:
    print("请安装 scipy: pip install scipy")
    exit(1)


# ================== Haar 小波滤波器 ==================
def haar_filters():
    lo = torch.tensor([1.0, 1.0]) / math.sqrt(2)
    hi = torch.tensor([1.0, -1.0]) / math.sqrt(2)
    LL = (lo.unsqueeze(0) * lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    LH = (lo.unsqueeze(0) * hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    HL = (hi.unsqueeze(0) * lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    HH = (hi.unsqueeze(0) * hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    return torch.cat([LL, LH, HL, HH], dim=0)


def dwt_haar(x, dec_filters):
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
    B, C, H, W = LL.shape
    coeffs = torch.stack([LL, LH, HL, HH], dim=2).reshape(B, C * 4, H, W)
    filters = rec_filters.repeat(C, 1, 1, 1)
    out = F.conv_transpose2d(coeffs, filters, stride=2, groups=C)
    return out[:, :, :2*H, :2*W]


# ================== db4 小波滤波器 ==================
def db4_filters():
    lo = torch.tensor([-0.010597401784997278, 0.032883011666982945, 0.030841381835986965,
                      -0.18703481171888114, -0.02798376941698385, 0.6308807679295904,
                       0.7148465705525415, 0.23037781330885523], dtype=torch.float32)
    hi = torch.tensor([-0.23037781330885523, 0.7148465705525415, -0.6308807679295904,
                      -0.02798376941698385, 0.18703481171888114, 0.030841381835986965,
                      -0.032883011666982945, -0.010597401784997278], dtype=torch.float32)
    return lo, hi


def make_2d_filters(lo, hi):
    LL = torch.outer(lo, lo).unsqueeze(0).unsqueeze(0)
    LH = torch.outer(lo, hi).unsqueeze(0).unsqueeze(0)
    HL = torch.outer(hi, lo).unsqueeze(0).unsqueeze(0)
    HH = torch.outer(hi, hi).unsqueeze(0).unsqueeze(0)
    return torch.cat([LL, LH, HL, HH], dim=0)


def dwt_db4(x, dec_filters):
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


def idwt_db4(LL, LH, HL, HH, rec_filters, orig_h, orig_w):
    B, C, H, W = LL.shape
    coeffs = torch.stack([LL, LH, HL, HH], dim=2).reshape(B, C * 4, H, W)
    filter_pad = rec_filters.shape[-1] // 2
    filters = rec_filters.repeat(C, 1, 1, 1)
    out = F.conv_transpose2d(coeffs, filters, stride=2, padding=filter_pad, groups=C)
    return out[:, :, :orig_h, :orig_w]


# ================== 通用软阈值与噪声估计 ==================
def soft_threshold_torch(coeff, lambd):
    return torch.where(coeff > lambd, coeff - lambd,
                       torch.where(coeff < -lambd, coeff + lambd, torch.zeros_like(coeff)))


def estimate_noise_sigma_torch(HH):
    hh_abs = torch.abs(HH.reshape(HH.shape[0], -1))
    median = torch.median(hh_abs, dim=1).values
    return median / 0.6745


# ================== 方法1: Haar DWT 去噪 ==================
def denoise_haar(img_tensor, threshold_factor=0.5):
    device = img_tensor.device
    dtype = img_tensor.dtype
    dec = haar_filters().to(device, dtype)
    rec = dec.clone()
    x_log = torch.log(torch.clamp(img_tensor, min=1e-6))
    LL, LH, HL, HH = dwt_haar(x_log, dec)
    sigma = estimate_noise_sigma_torch(HH)
    N = HH[0].numel()
    lambd = sigma * math.sqrt(2.0 * math.log(N)) * threshold_factor
    lambd = lambd.view(-1, 1, 1, 1)
    LH = soft_threshold_torch(LH, lambd)
    HL = soft_threshold_torch(HL, lambd)
    HH = soft_threshold_torch(HH, lambd)
    rec_log = idwt_haar(LL, LH, HL, HH, rec)
    out = torch.exp(rec_log)
    return out


# ================== 方法2: db4 DWT 去噪 (单级) ==================
def denoise_db4(img_tensor, threshold_factor=0.5):
    device = img_tensor.device
    dtype = img_tensor.dtype
    lo, hi = db4_filters()
    dec = make_2d_filters(lo, hi).to(device, dtype)
    rec = make_2d_filters(lo.flip(0), hi.flip(0)).to(device, dtype)
    x_log = torch.log(torch.clamp(img_tensor, min=1e-6))
    (LL, LH, HL, HH), orig_h, orig_w = dwt_db4(x_log, dec)
    sigma = estimate_noise_sigma_torch(HH)
    N = HH[0].numel()
    lambd = sigma * math.sqrt(2.0 * math.log(N)) * threshold_factor
    lambd = lambd.view(-1, 1, 1, 1)
    LH = soft_threshold_torch(LH, lambd)
    HL = soft_threshold_torch(HL, lambd)
    HH = soft_threshold_torch(HH, lambd)
    rec_log = idwt_db4(LL, LH, HL, HH, rec, orig_h, orig_w)
    out = torch.exp(rec_log)
    return out


# ================== 方法3: SWT 全局阈值去噪 ==================
def denoise_swt_global(img_np, wavelet='db4', threshold_factor=0.5):
    def swt2_one(img):
        coeffs = pywt.swt2(img, wavelet, level=1, start_level=0)
        return coeffs[0]
    # 逐通道
    out = np.zeros_like(img_np)
    for c in range(3):
        LL, (LH, HL, HH) = swt2_one(img_np[:,:,c])
        sigma = (np.median(np.abs(HH)) / 0.6745)
        N = HH.size
        lambd = sigma * math.sqrt(2.0 * math.log(N)) * threshold_factor
        def sth(x): return np.sign(x) * np.maximum(np.abs(x) - lambd, 0)
        LH_d, HL_d, HH_d = sth(LH), sth(HL), sth(HH)
        rec = pywt.iswt2((LL, (LH_d, HL_d, HH_d)), wavelet)
        out[:,:,c] = rec
    return out


# ================== 方法4: SWT 局部自适应阈值 (NeighShrink) ==================
def denoise_swt_local(img_np, wavelet='db4', window_size=7, factor=1.0):
    def swt2_one(img):
        coeffs = pywt.swt2(img, wavelet, level=1, start_level=0)
        return coeffs[0]
    out = np.zeros_like(img_np)
    for c in range(3):
        LL, (LH, HL, HH) = swt2_one(img_np[:,:,c])
        sigma = (np.median(np.abs(HH)) / 0.6745)
        def neigh_shrink(band):
            kernel = np.ones((window_size, window_size)) / (window_size ** 2)
            local_energy = scipy.ndimage.convolve(band ** 2, kernel, mode='reflect')
            S2 = np.maximum(local_energy - sigma ** 2, 0)
            shrink = S2 / (S2 + sigma ** 2 + 1e-8)
            return band * shrink * factor
        LH_d = neigh_shrink(LH)
        HL_d = neigh_shrink(HL)
        HH_d = neigh_shrink(HH)
        rec = pywt.iswt2((LL, (LH_d, HL_d, HH_d)), wavelet)
        out[:,:,c] = rec
    return out


# ================== 横向对比可视化 ==================
def compare_all_methods(
    input_dir=r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_YOLO\images\train",
    output_dir=r"./denoised_comparison_all",
    num_images=5,
    img_size=640,
    threshold_factor=0.5,
    wavelet='db4',
):
    os.makedirs(output_dir, exist_ok=True)
    all_imgs = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.bmp'))]
    selected = sorted(all_imgs)[:num_images]
    print(f"Comparing 4 methods on {len(selected)} images...")

    for idx, img_name in enumerate(selected):
        img_path = os.path.join(input_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((img_size, img_size), Image.BILINEAR)
        img_np = np.array(img).astype(np.float64) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).cuda()

        # 原始图像
        orig = img_np.copy()

        # 方法1: Haar DWT
        with torch.no_grad():
            haar_out = denoise_haar(img_tensor, threshold_factor)
        haar_np = haar_out.squeeze(0).permute(1,2,0).cpu().numpy()
        haar_np = np.clip(haar_np, 0, 1)

        # 方法2: db4 DWT
        with torch.no_grad():
            db4_out = denoise_db4(img_tensor, threshold_factor)
        db4_np = db4_out.squeeze(0).permute(1,2,0).cpu().numpy()
        db4_np = np.clip(db4_np, 0, 1)

        # 方法3: SWT 全局
        swt_global = denoise_swt_global(img_np, wavelet=wavelet, threshold_factor=threshold_factor)
        swt_global = np.clip(swt_global, 0, 1)

        # 方法4: SWT 局部
        swt_local = denoise_swt_local(img_np, wavelet=wavelet, window_size=7, factor=1.0)
        swt_local = np.clip(swt_local, 0, 1)

        # 绘图
        fig, axes = plt.subplots(1, 5, figsize=(30, 6))
        titles = ['Original', 'Haar DWT', 'db4 DWT', 'SWT Global', 'SWT Local']
        images = [orig, haar_np, db4_np, swt_global, swt_local]
        for ax, title, im in zip(axes, titles, images):
            ax.imshow(im)
            ax.set_title(title, fontsize=12)
            ax.axis('off')
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"compare_{os.path.splitext(img_name)[0]}.png")
        plt.savefig(save_path, dpi=100)
        plt.close()
        print(f"  Saved {save_path}")

    print("All comparisons completed. Output:", output_dir)


if __name__ == "__main__":
    compare_all_methods(
        input_dir=r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_YOLO\images\train",
        output_dir=r"./denoised_comparison_all",
        num_images=5,
        img_size=640,
        threshold_factor=0.5,
        wavelet='db4'
    )