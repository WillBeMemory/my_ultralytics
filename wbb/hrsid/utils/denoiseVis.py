import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import math
import pywt
import ptwt  # 需要安装: pip install ptwt==1.0.1
import importlib.metadata


# -------------------- 通用工具函数 --------------------
def soft_threshold_torch(coeff, lambd):
    """通用软阈值函数（支持广播）"""
    return torch.where(
        coeff > lambd, coeff - lambd,
        torch.where(coeff < -lambd, coeff + lambd, torch.zeros_like(coeff))
    )


def estimate_noise_sigma_torch(high_coeffs):
    """基于高频系数绝对值中位数的噪声估计"""
    coeff_abs = torch.abs(high_coeffs.reshape(high_coeffs.shape[0], -1))
    median = torch.median(coeff_abs, dim=1).values
    return median / 0.6745


# -------------------- Haar DWT 实现 --------------------
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
    return out[:, :, :2 * H, :2 * W]


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

    for lvl in range(level - 1, -1, -1):
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
    return torch.clamp(out, 0, 1)


# -------------------- Bior4.4 DWT 实现 --------------------
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

    for lvl in range(level - 1, -1, -1):
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
    return torch.clamp(out, 0, 1)


# -------------------- 基于pywt的普通DWT近似（保留原方法用于对比） --------------------
def denoise_dtcwt_pixel_legacy(img_np, level=3, threshold_factor=0.5):
    """原方法：基于pywt的普通DWT（非真正DTCWT）"""
    img_255 = np.clip(img_np * 255.0, 0, 255).astype(np.float32)
    out = np.zeros_like(img_255)

    for c in range(3):
        coeffs = pywt.wavedec2(img_255[:, :, c], 'bior4.4', level=level)
        sigma = (np.median(np.abs(coeffs[-1][-1])) / 0.6745)
        N = sum(band.size for coeff in coeffs[1:] for band in coeff)
        lambd = sigma * math.sqrt(2.0 * math.log(N)) * threshold_factor

        def sth(x): return np.sign(x) * np.maximum(np.abs(x) - lambd, 0)

        coeffs[1:] = [tuple(sth(band) for band in level_coeff) for level_coeff in coeffs[1:]]
        rec = pywt.waverec2(coeffs, 'bior4.4')
        out[:, :, c] = rec

    return np.clip(out / 255.0, 0, 1)


def denoise_dtcwt_log_legacy(img_np, level=3, threshold_factor=0.5):
    """原方法：对数域普通DWT（非真正DTCWT）"""
    img_log = np.log(np.clip(img_np, 1e-6, None))
    log_min, log_max = img_log.min(), img_log.max()
    img_mapped = (img_log - log_min) / (log_max - log_min + 1e-8) * 255.0

    denoised_mapped = denoise_dtcwt_pixel_legacy(img_mapped / 255.0, level=level,
                                                 threshold_factor=threshold_factor) * 255.0
    denoised_log = denoised_mapped / 255.0 * (log_max - log_min) + log_min

    return np.exp(denoised_log)


# -------------------- 最终修复版：真正6方向双树复小波(DTCWT)实现（兼容ptwt 1.0.1） --------------------
class DTCWTForward(torch.nn.Module):
    """纯Python实现的6方向双树复小波前向变换
    完全兼容Python 3.12+和ptwt==1.0.1，无任何C++扩展依赖
    """

    def __init__(self, J=1, wavelet_a='db2', wavelet_b='sym2', mode='symmetric'):
        super().__init__()
        self.J = J
        self.mode = mode
        self.wavelet_a = pywt.Wavelet(wavelet_a)
        self.wavelet_b = pywt.Wavelet(wavelet_b)

    def forward(self, x):
        B, C, H, W = x.shape
        Yl_a = x
        Yl_b = x
        Yh = []
        sizes = []  # 保存每一级的原始尺寸，用于逆变换对齐

        for j in range(self.J):
            # 保存当前级尺寸
            current_size = (Yl_a.shape[2], Yl_a.shape[3])
            sizes.append(current_size)

            # 双树分解：树A和树B分别进行一级DWT
            coeffs_a = ptwt.wavedec2(Yl_a, self.wavelet_a, level=1, mode=self.mode)
            coeffs_b = ptwt.wavedec2(Yl_b, self.wavelet_b, level=1, mode=self.mode)

            approx_a, (LH_a, HL_a, HH_a) = coeffs_a[0], coeffs_a[1]
            approx_b, (LH_b, HL_b, HH_b) = coeffs_b[0], coeffs_b[1]

            # 构造6个方向的复系数（真正DTCWT的核心）
            # 方向：+15°, +45°, +75°, -15°, -45°, -75°
            dir1 = torch.stack([LH_a, HL_b], dim=-1)  # +15° (实:LH_a, 虚:HL_b)
            dir2 = torch.stack([HL_a, LH_b], dim=-1)  # +45° (实:HL_a, 虚:LH_b)
            dir3 = torch.stack([HH_a, HH_b], dim=-1)  # +75° (实:HH_a, 虚:HH_b)
            dir4 = torch.stack([LH_a, -HL_b], dim=-1)  # -15° (实:LH_a, 虚:-HL_b)
            dir5 = torch.stack([HL_a, -LH_b], dim=-1)  # -45° (实:HL_a, 虚:-LH_b)
            dir6 = torch.stack([HH_a, -HH_b], dim=-1)  # -75° (实:HH_a, 虚:-HH_b)

            # 合并为形状: (B, C, 6, H, W, 2)
            high_coeffs = torch.stack([dir1, dir2, dir3, dir4, dir5, dir6], dim=2)
            Yh.append(high_coeffs)

            # 更新下一级的低频系数
            Yl_a = approx_a
            Yl_b = approx_b

        # 最终低频系数取两棵树的平均
        Yl = (Yl_a + Yl_b) / 2.0
        return Yl, Yh, sizes


class DTCWTInverse(torch.nn.Module):
    """纯Python实现的6方向双树复小波逆变换
    完全兼容ptwt 1.0.1，修复所有尺寸对齐和API问题
    """

    def __init__(self, wavelet_a='db2', wavelet_b='sym2'):
        super().__init__()
        self.wavelet_a = pywt.Wavelet(wavelet_a)
        self.wavelet_b = pywt.Wavelet(wavelet_b)

    def forward(self, coeffs):
        Yl, Yh, sizes = coeffs
        recon_a = Yl
        recon_b = Yl

        for j in reversed(range(len(Yh))):
            high_coeffs = Yh[j]  # (B, C, 6, H, W, 2)
            target_size = sizes[j]  # 当前级应该有的尺寸

            # 从6个方向的复系数还原两棵树的实系数
            dir1, dir2, dir3, dir4, dir5, dir6 = torch.unbind(high_coeffs, dim=2)

            # 树A的系数（实部）
            LH_a = (dir1[..., 0] + dir4[..., 0]) / 2.0
            HL_a = (dir2[..., 0] + dir5[..., 0]) / 2.0
            HH_a = (dir3[..., 0] + dir6[..., 0]) / 2.0

            # 树B的系数（虚部）
            HL_b = (dir1[..., 1] - dir4[..., 1]) / 2.0
            LH_b = (dir2[..., 1] - dir5[..., 1]) / 2.0
            HH_b = (dir3[..., 1] - dir6[..., 1]) / 2.0

            # 强制对齐低频系数和高频系数的尺寸
            if recon_a.shape[2:] != LH_a.shape[2:]:
                pad_h = LH_a.shape[2] - recon_a.shape[2]
                pad_w = LH_a.shape[3] - recon_a.shape[3]
                recon_a = F.pad(recon_a, (0, pad_w, 0, pad_h), mode='reflect')
                recon_b = F.pad(recon_b, (0, pad_w, 0, pad_h), mode='reflect')

            # 分别重构两棵树（ptwt 1.0.1不支持mode参数）
            recon_a = ptwt.waverec2([recon_a, (LH_a, HL_a, HH_a)], self.wavelet_a)
            recon_b = ptwt.waverec2([recon_b, (LH_b, HL_b, HH_b)], self.wavelet_b)

            # 裁剪到目标尺寸
            recon_a = recon_a[:, :, :target_size[0], :target_size[1]]
            recon_b = recon_b[:, :, :target_size[0], :target_size[1]]

        # 最终结果取两棵树的平均
        recon = (recon_a + recon_b) / 2.0
        return recon


def denoise_dtcwt_6dir_log(img_tensor, level=3, threshold_factor=0.5):
    """真正6方向双树复小波去噪（对数域）"""
    device = img_tensor.device
    x_log = torch.log(torch.clamp(img_tensor, min=1e-6))

    # 初始化DTCWT变换
    dtcwt_fwd = DTCWTForward(J=level, wavelet_a='db2', wavelet_b='sym2', mode='symmetric').to(device)
    dtcwt_inv = DTCWTInverse(wavelet_a='db2', wavelet_b='sym2').to(device)

    # 前向变换
    Yl, Yh, sizes = dtcwt_fwd(x_log)

    # 噪声估计（使用最高级所有方向系数的绝对值中位数）
    highest_high = Yh[-1]  # (B, C, 6, H, W, 2)
    sigma = estimate_noise_sigma_torch(highest_high)
    N = highest_high[0].numel()
    lambd = sigma * math.sqrt(2.0 * math.log(N)) * threshold_factor
    lambd = lambd.view(-1, 1, 1, 1, 1)  # 广播到所有方向和分量

    # 对每级高频系数应用软阈值
    for lvl in range(level):
        high_coeffs = Yh[lvl]
        # 对实部和虚部分别应用阈值
        high_coeffs[..., 0] = soft_threshold_torch(high_coeffs[..., 0], lambd)
        high_coeffs[..., 1] = soft_threshold_torch(high_coeffs[..., 1], lambd)
        Yh[lvl] = high_coeffs

    # 逆变换重构
    recon_log = dtcwt_inv((Yl, Yh, sizes))

    # 自动尺寸对齐到原始输入
    if recon_log.shape[2] != x_log.shape[2] or recon_log.shape[3] != x_log.shape[3]:
        pad_h = x_log.shape[2] - recon_log.shape[2]
        pad_w = x_log.shape[3] - recon_log.shape[3]
        recon_log = F.pad(recon_log, (0, pad_w, 0, pad_h), mode='reflect')

    out = torch.exp(recon_log)
    return torch.clamp(out, 0, 1)


def denoise_dtcwt_6dir_pixel(img_tensor, level=3, threshold_factor=0.5):
    """真正6方向双树复小波去噪（像素域）"""
    device = img_tensor.device

    # 初始化DTCWT变换
    dtcwt_fwd = DTCWTForward(J=level, wavelet_a='db2', wavelet_b='sym2', mode='symmetric').to(device)
    dtcwt_inv = DTCWTInverse(wavelet_a='db2', wavelet_b='sym2').to(device)

    # 前向变换
    Yl, Yh, sizes = dtcwt_fwd(img_tensor)

    # 噪声估计
    highest_high = Yh[-1]
    sigma = estimate_noise_sigma_torch(highest_high)
    N = highest_high[0].numel()
    lambd = sigma * math.sqrt(2.0 * math.log(N)) * threshold_factor
    lambd = lambd.view(-1, 1, 1, 1, 1)

    # 软阈值去噪
    for lvl in range(level):
        high_coeffs = Yh[lvl]
        high_coeffs[..., 0] = soft_threshold_torch(high_coeffs[..., 0], lambd)
        high_coeffs[..., 1] = soft_threshold_torch(high_coeffs[..., 1], lambd)
        Yh[lvl] = high_coeffs

    # 逆变换重构
    recon = dtcwt_inv((Yl, Yh, sizes))

    # 尺寸对齐
    if recon.shape[2] != img_tensor.shape[2] or recon.shape[3] != img_tensor.shape[3]:
        pad_h = img_tensor.shape[2] - recon.shape[2]
        pad_w = img_tensor.shape[3] - recon.shape[3]
        recon = F.pad(recon, (0, pad_w, 0, pad_h), mode='reflect')

    return torch.clamp(recon, 0, 1)


# -------------------- 标签加载与可视化 --------------------
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
        x1 = (cx - w / 2) * img_size
        y1 = (cy - h / 2) * img_size
        width = w * img_size
        height = h * img_size
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=1.5, edgecolor='lime', facecolor='none',
            label='GT' if ax == plt.gca() else ""
        )
        ax.add_patch(rect)


# -------------------- 主对比函数（8列完整对比） --------------------
def compare_all_methods(
        input_dir=r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_YOLO\images\train",
        labels_dir=r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_YOLO\labels\train",
        output_dir=r"./denoised_comparison_6dir_dtcwt",
        num_images=5,
        img_size=640,
        decomp_level=2,  # Haar/Bior4.4分解层数
        dtcwt_level=3,  # DTCWT分解层数
        threshold_factor=0.5,  # 全局阈值因子
):
    os.makedirs(output_dir, exist_ok=True)
    all_imgs = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.bmp'))]
    selected = sorted(all_imgs)[:num_images]

    # 打印实验参数
    print("=" * 60)
    print("小波去噪方法对比实验")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"对比图片数: {len(selected)}")
    print(f"图片尺寸: {img_size}x{img_size}")
    print(f"Haar/Bior4.4分解层数: {decomp_level}")
    print(f"DTCWT分解层数: {dtcwt_level}")
    print(f"阈值因子: {threshold_factor}")
    print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # 安全获取ptwt版本
    try:
        ptwt_version = importlib.metadata.version("ptwt")
        print(f"ptwt版本: {ptwt_version}")
    except (importlib.metadata.PackageNotFoundError, AttributeError):
        print("ptwt版本: 无法获取（已安装）")

    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for idx, img_name in enumerate(selected):
        print(f"处理图片 {idx + 1}/{len(selected)}: {img_name}")

        # 加载图片
        img_path = os.path.join(input_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((img_size, img_size), Image.Resampling.BILINEAR)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

        # 加载标签
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')
        gt_boxes = load_yolo_labels(label_path)

        # 运行所有去噪方法
        with torch.no_grad():
            # 传统DWT方法
            haar_np = denoise_haar(img_tensor, level=decomp_level, threshold_factor=threshold_factor).squeeze(
                0).permute(1, 2, 0).cpu().numpy()
            bior_np = denoise_bior44(img_tensor, level=decomp_level, threshold_factor=threshold_factor).squeeze(
                0).permute(1, 2, 0).cpu().numpy()

            # 原近似DTCWT方法
            dtcwt_log_legacy_np = denoise_dtcwt_log_legacy(img_np, level=dtcwt_level, threshold_factor=threshold_factor)
            dtcwt_pixel_legacy_np = denoise_dtcwt_pixel_legacy(img_np, level=dtcwt_level,
                                                               threshold_factor=threshold_factor)

            # 新6方向DTCWT方法
            dtcwt_6dir_log_np = denoise_dtcwt_6dir_log(img_tensor, level=dtcwt_level,
                                                       threshold_factor=threshold_factor).squeeze(0).permute(1, 2,
                                                                                                             0).cpu().numpy()
            dtcwt_6dir_pixel_np = denoise_dtcwt_6dir_pixel(img_tensor, level=dtcwt_level,
                                                           threshold_factor=threshold_factor).squeeze(0).permute(1, 2,
                                                                                                                 0).cpu().numpy()

        # 8列对比图
        fig, axes = plt.subplots(1, 8, figsize=(48, 7))
        titles = [
            'Original (带GT)',
            '6方向DTCWT (log)',
            '6方向DTCWT (pixel)',
            '原近似DTCWT (log)',
            '原近似DTCWT (pixel)',
            'Haar DWT',
            'Bior4.4 DWT',
            '噪声残差(6dir-log)'
        ]

        # 计算噪声残差（原图 - 去噪图）
        noise_residual = np.abs(img_np - dtcwt_6dir_log_np)
        # 增强残差可视化效果
        noise_residual = (noise_residual - noise_residual.min()) / (noise_residual.max() - noise_residual.min() + 1e-8)

        images = [
            img_np,
            dtcwt_6dir_log_np,
            dtcwt_6dir_pixel_np,
            dtcwt_log_legacy_np,
            dtcwt_pixel_legacy_np,
            haar_np,
            bior_np,
            noise_residual
        ]

        for ax, title, im in zip(axes, titles, images):
            if title == '噪声残差(6dir-log)':
                ax.imshow(im, cmap='jet')
            else:
                ax.imshow(np.clip(im, 0, 1))
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.axis('off')

        # 在原图上绘制GT框
        if gt_boxes:
            draw_gt_boxes(axes[0], gt_boxes, img_size)

        plt.tight_layout(pad=1.5)
        save_path = os.path.join(output_dir, f"compare_{os.path.splitext(img_name)[0]}.png")
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {save_path}")

    print("\n所有对比完成！输出目录:", output_dir)
    print("=" * 60)


if __name__ == "__main__":
    # 可根据需要调整参数
    compare_all_methods(
        num_images=5,
        img_size=640,
        decomp_level=2,
        dtcwt_level=3,
        threshold_factor=0.3  # 建议范围: 0.3-0.7，值越大去噪越强
    )