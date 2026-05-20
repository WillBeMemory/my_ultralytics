import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from ultralytics.nn.modules.AddModules.LogWaveletDenoise import LogWaveletDenoise  # 请根据实际路径调整


def visualize_denoising(
    input_dir: str = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_YOLO\images\train",
    output_dir: str = r"./log_wavelet_denoise_vis",
    num_images: int = 50,
    img_size: int = 640,
    threshold_factor: float = 0.3,
    level: int = 1,
):
    """
    使用 LogWaveletDenoise 模块对图像去噪，并保存原图与去噪图的并排对比。
    参数:
        input_dir:         原始图像文件夹
        output_dir:        对比图输出文件夹
        num_images:        处理的图像数量（默认50）
        img_size:          统一调整尺寸（640x640）
        threshold_factor:  阈值缩放因子（0~1，越大去噪越强）
        level:             小波分解级数（默认1）
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)

    # 加载去噪模块（不下采样，仅去噪）
    denoiser = LogWaveletDenoise(
        c1=3, c2=3,                      # 输入输出均为3通道RGB
        threshold_factor=threshold_factor,
        level=level,
        downsample=False                 # 关键：保持原尺寸
    ).to(device)
    denoiser.eval()

    # 获取图像列表（按名称排序，取前 num_images 张）
    all_imgs = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.bmp'))]
    selected = sorted(all_imgs)[:num_images]
    print(f"Processing {len(selected)} images with threshold_factor={threshold_factor}, level={level}...")

    for idx, img_name in enumerate(selected):
        img_path = os.path.join(input_dir, img_name)

        # 读取图像 -> RGB -> resize -> 归一化到 [0,1]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((img_size, img_size), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32) / 255.0

        # 转为张量 (1, 3, H, W)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

        # 去噪
        with torch.no_grad():
            denoised_tensor = denoiser(img_tensor)

        # 转回 numpy 并裁剪到 [0, 1]
        denoised_np = denoised_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        denoised_np = np.clip(denoised_np, 0, 1)

        # 生成并排对比图
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img_np)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(denoised_np)
        axes[1].set_title(f"Denoised (th={threshold_factor}, level={level})")
        axes[1].axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"compare_{os.path.splitext(img_name)[0]}.png")
        plt.savefig(save_path, dpi=100)
        plt.close()

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx+1}/{len(selected)} images")

    print("Visualization completed. Output saved to:", output_dir)


if __name__ == "__main__":
    # 示例：阈值0.3，单级分解
    visualize_denoising(
        input_dir=r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_YOLO\images\train",
        output_dir=r"./log_wavelet_denoise_vis",
        num_images=10,            # 可先测试10张
        threshold_factor=0.3,
        level=2
    )