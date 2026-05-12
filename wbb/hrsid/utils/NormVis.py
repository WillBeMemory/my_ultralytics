import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
import random
from ultralytics import YOLO

# ================== 参数配置 ==================
model_path = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\hrsid\pt\yolo11n-wavelet-hipa-test-7c46e-5090d.pt"
dataset_root = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_YOLO"
output_dir = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\hrsid\simulate\norm_visualization"
num_samples = 50
img_size = 640

os.makedirs(output_dir, exist_ok=True)

# ================== 数据集工具函数 ==================
def load_yolo_labels(label_path, img_w, img_h):
    targets = []
    if not os.path.exists(label_path):
        return targets
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, cx, cy, w, h = map(float, parts)
            targets.append((cx, cy, w, h))
    return targets

# ================== 固定 Backbone（截取第 8 层 C3k2 后的 P5） ==================
class PretrainedBackbone(nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        yolo = YOLO(model_path)
        # 仅取前 8 层（索引 0 ~ 7），对应 C3k2 处理后的 P5 特征图
        num_backbone = 10
        backbone_layers = list(yolo.model.model.children())[:num_backbone]
        self.backbone = nn.Sequential(*backbone_layers)

        # 冻结参数
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.to(device)
        self.device = device

    def forward(self, x):
        with torch.no_grad():
            return self.backbone(x)   # 输出 (1, C, 20, 20)

# ================== 主程序 ==================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    backbone = PretrainedBackbone(model_path, device)

    train_img_dir = os.path.join(dataset_root, 'images', 'train')
    train_label_dir = os.path.join(dataset_root, 'labels', 'train')
    all_imgs = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.png', '.bmp'))]
    selected_imgs = random.sample(all_imgs, min(num_samples, len(all_imgs)))

    for img_name in selected_imgs:
        img_path = os.path.join(train_img_dir, img_name)
        label_path = os.path.join(train_label_dir, os.path.splitext(img_name)[0] + '.txt')

        # 加载图像并预处理
        img = Image.open(img_path).convert('RGB')
        img = img.resize((img_size, img_size), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

        targets = load_yolo_labels(label_path, img.width, img.height)

        # 提取第 8 层 P5 特征图
        p5 = backbone(img_tensor)
        print(f"P5 feature shape (from layer 8 C3k2): {p5.shape}")

        # ---- 通道均值图 ----
        p5_mean_map = p5.mean(dim=1, keepdim=False)

        # ---- L1 和 L2 范数图 ----
        l1_map = p5.abs().sum(dim=1, keepdim=False)
        l2_map = torch.norm(p5, p=2, dim=1, keepdim=False)

        # ---- 激活稀疏度图（每个位置活跃的通道数） ----
        thresh = p5.abs().mean() * 1.5  # 动态阈值，可按需改为固定值如 0.5
        active_mask = (p5.abs() > thresh).float()
        sparsity_map = active_mask.sum(dim=1)  # (1,20,20) 值越小越稀疏

        # ---- 目标-背景对比度图（基于 L1 范数） ----
        # 建立 20×20 的 GT 掩膜
        mask = torch.zeros((1, 20, 20), device=device)
        for cx, cy, w, h in targets:
            x1 = int((cx - w/2) * 20)
            y1 = int((cy - h/2) * 20)
            x2 = int((cx + w/2) * 20 + 1)
            y2 = int((cy + h/2) * 20 + 1)
            # 边界裁剪
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(20, x2), min(20, y2)
            mask[0, y1:y2, x1:x2] = 1.0

        # 背景平均 L1
        bg_pixels = (1 - mask).sum() + 1e-6
        bg_mean_l1 = (l1_map * (1 - mask)).sum() / bg_pixels

        if mask.sum() > 0:
            contrast_map = l1_map / (bg_mean_l1 + 1e-6)
            contrast_map = contrast_map * mask  # 只保留目标区域的对比度
        else:
            contrast_map = torch.zeros_like(l1_map)

        # ---- 归一化并上采样到 640×640 ----
        def normalize_and_resize(tensor_map, size):
            arr = tensor_map.cpu().numpy().squeeze()
            vmin, vmax = arr.min(), arr.max()
            if vmax - vmin > 1e-6:
                arr = (arr - vmin) / (vmax - vmin)
            else:
                arr = np.zeros_like(arr)
            arr = (arr * 255).astype(np.uint8)
            img_pil = Image.fromarray(arr).resize((size, size), Image.NEAREST)
            return np.array(img_pil) / 255.0

        p5_mean_large = normalize_and_resize(p5_mean_map, img_size)
        l1_large      = normalize_and_resize(l1_map, img_size)
        l2_large      = normalize_and_resize(l2_map, img_size)
        sparsity_large = normalize_and_resize(sparsity_map, img_size)
        contrast_large = normalize_and_resize(contrast_map, img_size)

        # ---- 绘制原图 + GT 框 ----
        img_draw = Image.fromarray((img_np * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img_draw)
        for cx, cy, w, h in targets:
            x1 = (cx - w/2) * img_size
            y1 = (cy - h/2) * img_size
            x2 = (cx + w/2) * img_size
            y2 = (cy + h/2) * img_size
            draw.rectangle([x1, y1, x2, y2], outline='lime', width=2)

        # ---- 2×3 可视化 ----
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        axes[0].imshow(img_draw)
        axes[0].set_title('Original + GT')
        axes[0].axis('off')

        axes[1].imshow(p5_mean_large, cmap='gray')
        axes[1].set_title('P5 Mean (C-dim avg)')
        axes[1].axis('off')

        axes[2].imshow(l1_large, cmap='gray')
        axes[2].set_title('L1 Norm')
        axes[2].axis('off')

        axes[3].imshow(l2_large, cmap='gray')
        axes[3].set_title('L2 Norm')
        axes[3].axis('off')

        axes[4].imshow(sparsity_large, cmap='hot')
        axes[4].set_title('Active Channels (> th)')
        axes[4].axis('off')

        axes[5].imshow(contrast_large, cmap='plasma')
        axes[5].set_title('Object Contrast (L1/BG)')
        axes[5].axis('off')

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"norm_{os.path.splitext(img_name)[0]}.png")
        plt.savefig(out_path, dpi=100)
        plt.close()
        print(f"Saved {out_path}")

    print("All visualizations completed.")

if __name__ == "__main__":
    main()