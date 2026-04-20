import numpy as np
import cv2
import matplotlib.pyplot as plt
import pywt
from skimage import img_as_float, img_as_ubyte


def wavelet_despeckle(image_path, wavelet='db4', level=2, threshold_scale=0.5, save_path=None):
    """
    小波阈值去斑（针对SAR图像的乘性噪声）
    参数：
        image_path: 输入图像路径（支持jpg/png等格式）
        wavelet:    小波基，推荐 'db4' 或 'sym4'
        level:      分解层数，通常2-3层
        threshold_scale: 阈值缩放因子，0.3~0.7之间，越大去斑越强但可能损失细节
        save_path:  如果提供，保存去斑后的图像到该路径
    返回：
        denoised:  去斑后的图像（uint8格式）
    """
    # 1. 读取图像（转为灰度图）
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")
    original = img.copy()

    # 2. 转换为浮点数并归一化到 [0,1]
    img_float = img_as_float(img)  # 取值范围 [0,1]

    # 3. 对数变换：将乘性噪声转换为加性噪声
    #    加一个小常数避免 log(0)
    log_img = np.log1p(img_float * 255) / 255.0  # 保持范围近似 [0,1]

    # 4. 小波分解
    coeffs = pywt.wavedec2(log_img, wavelet, level=level)

    # 5. 估计噪声标准差（使用第一级高频子带的中位数绝对偏差）
    #    高频子带通常包含三个方向：cH, cV, cD
    #    这里取第一个方向 cH 作为估计
    sigma = np.median(np.abs(coeffs[-1][0])) / 0.6745
    threshold = sigma * threshold_scale

    # 6. 对每一层的高频子带进行软阈值处理
    coeffs_thresh = list(coeffs)
    for i in range(1, len(coeffs)):  # 跳过近似系数（索引0）
        # coeffs[i] 是元组 (cH, cV, cD)
        thresh_tuple = tuple(
            pywt.threshold(c, threshold, mode='soft') for c in coeffs[i]
        )
        coeffs_thresh[i] = thresh_tuple

    # 7. 小波重构
    log_denoised = pywt.waverec2(coeffs_thresh, wavelet)

    # 8. 指数变换还原
    #    注意：重构后的图像尺寸可能与原始略有差异（因为边界效应），需要裁剪
    h, w = original.shape
    log_denoised = log_denoised[:h, :w]
    denoised_float = np.expm1(log_denoised * 255) / 255.0
    denoised_float = np.clip(denoised_float, 0, 1)

    # 9. 转换为 uint8
    denoised = img_as_ubyte(denoised_float)

    # 10. 可选保存结果
    if save_path:
        cv2.imwrite(save_path, denoised)

    return original, denoised


# ================== 使用示例 ==================
if __name__ == "__main__":
    input_image = "D:/Study/PostGraduate/YOLO/datasets/SSDD/images/val/000015.jpg"
    output_image = "D:/Study/PostGraduate/YOLO/datasets/SSDD/images_val_denoised/000015.jpg"

    original, denoised = wavelet_despeckle(
        image_path=input_image,
        wavelet='db4',
        level=2,
        threshold_scale=0.4,
        save_path=output_image
    )

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(denoised, cmap='gray')
    plt.title('Wavelet Denoised')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    print(f"Denoised image saved to: {output_image}")