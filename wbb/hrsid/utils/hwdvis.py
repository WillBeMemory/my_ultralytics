import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from ultralytics.nn.modules import HWD


# ================== 小波分解与重建（来自您验证正确的代码） ==================
def haar_filters_groups(in_channels: int):
    dec_lo = torch.tensor([1/2, 1/2])
    dec_hi = torch.tensor([1/2, -1/2])
    base_LL = (dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    base_LH = (dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    base_HL = (dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    base_HH = (dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    per_channel = torch.cat([base_LL, base_LH, base_HL, base_HH], dim=1)  # (1,4,2,2)
    per_channel = per_channel.repeat(in_channels, 1, 1, 1)
    return per_channel.reshape(in_channels * 4, 1, 2, 2)

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

# ================== 改进的显示函数（逐通道独立拉伸） ==================
def tensor_to_gray_stretched(tensor):
    """逐通道独立拉伸到 [0,1] 再取平均，保留方向纹理"""
    img = tensor.squeeze(0).detach().cpu().numpy()   # (C, H, W)
    stretched = np.zeros_like(img)
    for c in range(img.shape[0]):
        ch = img[c]
        vmin, vmax = ch.min(), ch.max()
        if vmax - vmin > 1e-6:
            stretched[c] = (ch - vmin) / (vmax - vmin)
        else:
            stretched[c] = 0.5
    # 平均各通道并转为 uint8
    gray = stretched.mean(axis=0)
    return (gray * 255).astype(np.uint8)

def show_image(ax, img, title):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

# ================== 主测试 ==================
def hwd_visual(image_path, c1=3, c2=64):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 加载图像并转为 tensor (1, c1, H, W)
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    # 创建 HWD 模块（用于获取融合特征图）
    hwd = HWD(c1=c1, c2=c2).to(device)
    hwd.eval()

    # 填充到偶数尺寸
    B, C, H, W = img_tensor.shape
    pad_h = (2 - H % 2) % 2
    pad_w = (2 - W % 2) % 2
    x_padded = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect') if pad_h or pad_w else img_tensor

    with torch.no_grad():
        # ---------- 用刚才验证正确的函数进行小波分解 ----------
        LL, LH, HL, HH = wt_decompose_level(x_padded)

        # 转为灰度图（逐通道拉伸）
        ll_img = tensor_to_gray_stretched(LL)
        lh_img = tensor_to_gray_stretched(LH)
        hl_img = tensor_to_gray_stretched(HL)
        hh_img = tensor_to_gray_stretched(HH)

        # ---------- 融合后的 HWD 特征图 ----------
        fused_feat = hwd(img_tensor)          # (1, c2, H/2, W/2)
        fused_img = tensor_to_gray_stretched(fused_feat)

    # ---------- 绘图 ----------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    show_image(axes[0, 0], ll_img, 'LL (Low-frequency)')
    show_image(axes[0, 1], lh_img, 'LH (Horizontal edges)')
    show_image(axes[0, 2], hl_img, 'HL (Vertical edges)')
    show_image(axes[1, 0], hh_img, 'HH (Diagonal)')
    show_image(axes[1, 1], fused_img, 'Fused Feature Map (HWD output)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('hwd_visualization.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    image_path = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\vis\P0001_0_800_7200_8000.jpg"
    hwd_visual(image_path, c1=3, c2=64)