import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import random
from ultralytics import YOLO

# ================== 参数配置 ==================
model_path = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\hrsid\pt\yolo11s-test-44729c-5090d.pt"
model_path = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\hrsid\pt\yolo11n-t10.pt"
dataset_root = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_COMPLEX"
output_dir = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\hrsid\simulate\fpn_visualization_modules"
num_samples = 10
img_size = 640
mode = 'first'  # 'random' 或 'first'

os.makedirs(output_dir, exist_ok=True)

# ================== 数据集工具函数 ==================
def load_yolo_labels(label_path):
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

# ================== 顶层模块特征提取器（使用类名） ==================
class TopModuleFeatureExtractor(nn.Module):
    """提取 YOLO 模型顶层模块的输出特征图，并记录类名（例如 Conv, C3k2, SPPF）"""
    def __init__(self, model_path, device):
        super().__init__()
        yolo = YOLO(model_path, verbose=False)
        self.model = yolo.model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(device)
        self.device = device

        self.module_features = []  # list of (module_name, tensor)
        self.hooks = []

        # 遍历顶层子模块，获取索引和模块实例，使用类名
        for idx, module in enumerate(self.model.model.children()):
            class_name = module.__class__.__name__   # 例如 'Conv', 'C3k2', 'SPPF', 'Upsample', 'Concat', 'Detect'
            unique_name = f"{idx}:{class_name}"       # 例如 "0:Conv", "2:C3k2"
            # 使用工厂函数避免闭包延迟绑定
            def make_hook(name):
                def hook_fn(module, input, output):
                    if isinstance(output, (list, tuple)):
                        out = output[0]
                    else:
                        out = output
                    if isinstance(out, torch.Tensor) and len(out.shape) == 4:
                        _, _, H, W = out.shape
                        if H == W and H in [20, 40, 80, 160]:
                            self.module_features.append((name, out.detach()))
                return hook_fn
            self.hooks.append(module.register_forward_hook(make_hook(unique_name)))

    def forward(self, x):
        self.module_features = []
        with torch.no_grad():
            _ = self.model(x)
        return self.module_features

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

# ================== 归一化 & 上采样 ==================
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

# ================== 紧凑型可视化：4 行 × N 列 ==================
def plot_top_modules(img_draw, module_features, targets, img_size, output_path, max_cols=40):
    if len(module_features) > max_cols:
        print(f"Warning: {len(module_features)} modules found, only first {max_cols} will be shown.")
        module_features = module_features[:max_cols]

    n_cols = len(module_features) + 1  # +1 原图
    # 图形尺寸进一步压缩
    fig = plt.figure(figsize=(n_cols * 2.0, 12))
    gs = gridspec.GridSpec(4, n_cols, figure=fig,
                           width_ratios=[1.8] + [1.0] * (n_cols - 1),
                           wspace=0.04, hspace=0.06)   # 更小的间距

    # 原图（第 0 列，跨 4 行）
    ax_img = fig.add_subplot(gs[0:4, 0])
    ax_img.imshow(img_draw)
    ax_img.set_title('Original with GT', fontsize=14, fontweight='bold', pad=6)
    ax_img.axis('off')

    row_titles = ['Mean', 'L1', 'ObjContrast', 'FullContrast']
    row_cmaps  = ['gray', 'gray', 'plasma', 'plasma']

    for col_idx, (name, feat) in enumerate(module_features):
        col_pos = col_idx + 1
        _, C, H, W = feat.shape

        # 生成 GT 掩膜
        mask = torch.zeros((1, H, W), device=feat.device)
        for cx, cy, w, h in targets:
            x1 = int((cx - w/2) * W)
            y1 = int((cy - h/2) * H)
            x2 = int((cx + w/2) * W + 1)
            y2 = int((cy + h/2) * H + 1)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            mask[0, y1:y2, x1:x2] = 1.0

        mean_map = feat.mean(dim=1)
        l1_map = feat.abs().sum(dim=1)
        bg_mean = (l1_map * (1 - mask)).sum() / ((1 - mask).sum() + 1e-6)
        full_contrast = l1_map / (bg_mean + 1e-6)
        obj_contrast = full_contrast * mask

        mean_large = normalize_and_resize(mean_map, img_size)
        l1_large   = normalize_and_resize(l1_map, img_size)
        obj_large  = normalize_and_resize(obj_contrast, img_size)
        full_large = normalize_and_resize(full_contrast, img_size)

        # 统一标题，显示模块名和特征类型
        ax1 = fig.add_subplot(gs[0, col_pos])
        ax1.imshow(mean_large, cmap='gray')
        ax1.set_title(f'{name}\n{row_titles[0]}\n({H}×{W})', fontsize=8, pad=2)
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[1, col_pos])
        ax2.imshow(l1_large, cmap='gray')
        ax2.set_title(f'{name}\n{row_titles[1]}', fontsize=8, pad=2)
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[2, col_pos])
        ax3.imshow(obj_large, cmap='plasma')
        ax3.set_title(f'{name}\n{row_titles[2]}', fontsize=8, pad=2)
        ax3.axis('off')

        ax4 = fig.add_subplot(gs[3, col_pos])
        ax4.imshow(full_large, cmap='plasma')
        ax4.set_title(f'{name}\n{row_titles[3]}', fontsize=8, pad=2)
        ax4.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

# ================== 主程序 ==================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    extractor = TopModuleFeatureExtractor(model_path, device)

    train_img_dir = os.path.join(dataset_root, 'images', 'train')
    train_label_dir = os.path.join(dataset_root, 'labels', 'train')
    all_imgs = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.png', '.bmp'))]

    if mode == 'first':
        selected_imgs = all_imgs[:min(num_samples, len(all_imgs))]
    elif mode == 'random':
        selected_imgs = random.sample(all_imgs, min(num_samples, len(all_imgs)))
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'random' or 'first'.")

    for img_name in selected_imgs:
        img_path = os.path.join(train_img_dir, img_name)
        label_path = os.path.join(train_label_dir, os.path.splitext(img_name)[0] + '.txt')

        img = Image.open(img_path).convert('RGB')
        img = img.resize((img_size, img_size), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

        targets = load_yolo_labels(label_path)

        img_draw = Image.fromarray((img_np * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img_draw)
        for cx, cy, w, h in targets:
            x1 = (cx - w/2) * img_size
            y1 = (cy - h/2) * img_size
            x2 = (cx + w/2) * img_size
            y2 = (cy + h/2) * img_size
            draw.rectangle([x1, y1, x2, y2], outline='lime', width=2)

        module_features = extractor(img_tensor)
        print(f"Processing {img_name}: {len(module_features)} modules captured.")

        out_path = os.path.join(output_dir, f"modules_{os.path.splitext(img_name)[0]}.png")
        plot_top_modules(img_draw, module_features, targets, img_size, out_path, max_cols=40)

    extractor.remove_hooks()
    print("All visualizations completed.")

if __name__ == "__main__":
    main()