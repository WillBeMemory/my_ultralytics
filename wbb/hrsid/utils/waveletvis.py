import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---------- 正确的 Haar 滤波器生成 (groups=C 专用) ----------
def haar_filters_groups(in_channels: int):
    """
    返回形状 (in_channels*4, 1, 2, 2)，严格保证每组内顺序：LL, LH, HL, HH
    """
    dec_lo = torch.tensor([1/2, 1/2])
    dec_hi = torch.tensor([1/2, -1/2])
    # 四个基滤波器，每个形状 (1,1,2,2)
    base_LL = (dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    base_LH = (dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    base_HL = (dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    base_HH = (dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    # 沿 dim=1 拼接得到 (1,4,2,2)，表示一个通道的四个滤波器
    per_channel = torch.cat([base_LL, base_LH, base_HL, base_HH], dim=1)  # (1,4,2,2)
    # 重复 C 次，得到 (C,4,2,2)
    per_channel = per_channel.repeat(in_channels, 1, 1, 1)
    # 压平成 (4C, 1, 2, 2)，满足 groups=C 的排列要求
    return per_channel.reshape(in_channels * 4, 1, 2, 2)

# ---------- 单级 Haar 小波分解 (groups=C) ----------
def wt_decompose_level(x: torch.Tensor):
    """输入 (B, C, H, W)，返回 LL, LH, HL, HH 尺寸均为 (B, C, H/2, W/2)"""
    B, C, H, W = x.shape
    filters = haar_filters_groups(C).to(x.device)
    coeffs = F.conv2d(x, filters, stride=2, groups=C, padding=0)  # (B, 4C, H/2, W/2)
    coeffs = coeffs.view(B, C, 4, H//2, W//2)
    LL = coeffs[:, :, 0, :, :]
    LH = coeffs[:, :, 1, :, :]
    HL = coeffs[:, :, 2, :, :]
    HH = coeffs[:, :, 3, :, :]
    return LL, LH, HL, HH

# ---------- 单级逆 Haar 小波重建 ----------
def wt_reconstruct_level(LL: torch.Tensor, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor):
    """输入四个子带 (B, C, H/2, W/2)，返回重建图像 (B, C, H, W)"""
    a = LL + LH + HL + HH
    b = LL + LH - HL - HH
    c = LL - LH + HL - HH
    d = LL - LH - HL + HH
    B, C, H2, W2 = LL.shape
    combined = torch.stack([a, b, c, d], dim=2)       # (B, C, 4, H2, W2)
    combined = combined.reshape(B, C * 4, H2, W2)     # (B, 4C, H2, W2)
    return F.pixel_shuffle(combined, 2)               # (B, C, H, W)

# ---------- 显示辅助函数 ----------
def tensor_to_gray_stretched(tensor):
    """子带显示：拉伸对比度至 0-255"""
    img = tensor.squeeze(0).mean(dim=0).detach().cpu().numpy()
    vmin, vmax = img.min(), img.max()
    img = (img - vmin) / (vmax - vmin + 1e-6)
    return (img * 255).astype(np.uint8)

def tensor_to_gray_preserve(tensor):
    """重建显示：保持0-1浮点，matplotlib自动拉伸"""
    img = tensor.squeeze(0).mean(dim=0).detach().cpu().numpy()
    return np.clip(img, 0.0, 1.0)

# ---------- 主程序 ----------
def main():
    image_path = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\vis\P0001_0_800_7200_8000.jpg"
    # 读取图像并转为 tensor (1, 3, H, W)
    img = Image.open(image_path).convert('RGB')
    orig_h, orig_w = img.height, img.width
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    # 填充到偶数尺寸
    H, W = img_tensor.shape[2:]
    pad_h = (2 - H % 2) % 2
    pad_w = (2 - W % 2) % 2
    if pad_h or pad_w:
        img_padded = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    else:
        img_padded = img_tensor

    # 小波分解（groups=C 正确版）
    LL, LH, HL, HH = wt_decompose_level(img_padded)

    # 生成子带灰度图
    ll_img = tensor_to_gray_stretched(LL)
    lh_img = tensor_to_gray_stretched(LH)
    hl_img = tensor_to_gray_stretched(HL)
    hh_img = tensor_to_gray_stretched(HH)

    # 逆变换重建
    restored = wt_reconstruct_level(LL, LH, HL, HH)    # (1,3,H_pad,W_pad)
    restored = restored[:, :, :orig_h, :orig_w]        # 裁剪填充区域

    # 计算重建误差
    diff = (restored - img_tensor).abs().max().item()
    print(f"最大重建误差: {diff:.6f}")

    # 重建灰度图
    restored_gray = tensor_to_gray_preserve(restored)

    # 绘图显示
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0,0].imshow(ll_img, cmap='gray')
    axes[0,0].set_title('LL')
    axes[0,1].imshow(lh_img, cmap='gray')
    axes[0,1].set_title('LH')
    axes[0,2].imshow(hl_img, cmap='gray')
    axes[0,2].set_title('HL')
    axes[1,0].imshow(hh_img, cmap='gray')
    axes[1,0].set_title('HH')
    axes[1,1].imshow(restored_gray, cmap='gray')
    axes[1,1].set_title('Reconstructed')
    axes[1,2].axis('off')
    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()