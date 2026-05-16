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
# model_path = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\hrsid\pt\yolo11n-wavelet-hipa-test-7c46e-5090d.pt"
# model_path = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\hrsid\pt\yolo11n-wavelet-hipa-test-43513-5070ti.pt"
# model_path = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\hrsid\pt\yolo11n-wavelet-hipa-test-56c09-5090d.pt"
model_path = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\hrsid\pt\yolo11n-wavelet-hipa-test-36c82-5090d.pt"
# model_path = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\hrsid\pt\yolo11n-t10.pt"
# model_path = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\hrsid\pt\yolo11n-wavelet-hipa-test-6f1a4f-5070ti.pt"
dataset_root = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_COMPLEX"
output_dir = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\hrsid\simulate\fpn_visualization"
num_samples = 10
img_size = 640
mode = 'first'  # 可选 'random' 或 'first'，决定选取图片的方式

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

# ================== 特征提取器（新增 PAN 层捕获） ==================
class FPNFeatureExtractor(nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        # 使用 verbose=False 禁止终端进度条，避免 iShellPro 换行问题
        yolo = YOLO(model_path, verbose=False)
        self.model = yolo.model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(device)
        self.device = device

        self.feature_sequence = {}   # {size: [tensor, ...]}
        self.named_features = {}     # {'sppf': tensor, 'c2psa': tensor}
        self.hooks = []

        def hook_fn(module, input, output):
            if isinstance(output, (list, tuple)):
                out = output[0]
            else:
                out = output
            if isinstance(out, torch.Tensor) and len(out.shape) == 4:
                _, _, H, W = out.shape
                if H == W:
                    if H not in self.feature_sequence:
                        self.feature_sequence[H] = []
                    self.feature_sequence[H].append(out.detach())

            # 捕获 SPPF 和 C2PSA（按模块名称）
            for key in ['sppf', 'c2psa']:
                if key in module.__class__.__name__.lower():
                    self.named_features[key] = out.detach()

        for module in self.model.modules():
            self.hooks.append(module.register_forward_hook(hook_fn))

    def forward(self, x):
        self.feature_sequence = {}
        self.named_features = {}
        with torch.no_grad():
            _ = self.model(x)

        def get_first(size):
            if size in self.feature_sequence and len(self.feature_sequence[size]) >= 1:
                return self.feature_sequence[size][0]
            return None

        def get_second(size):
            if size in self.feature_sequence and len(self.feature_sequence[size]) >= 2:
                return self.feature_sequence[size][1]
            return None

        def get_third(size):
            if size in self.feature_sequence and len(self.feature_sequence[size]) >= 3:
                return self.feature_sequence[size][2]
            return None

        # Backbone
        p3_backbone = get_first(80)
        p4_backbone = get_first(40)
        p5_backbone = get_first(20)

        # FPN（第二次出现）
        p4_fpn = get_second(40)
        p3_fpn = get_second(80)

        # PAN（第三次出现）
        p4_pan = get_third(40)
        p5_pan = get_third(20)

        # SPPF / C2PSA
        sppf_feat = self.named_features.get('sppf', None)
        c2psa_feat = self.named_features.get('c2psa', None)

        return {
            'P3_backbone': p3_backbone,
            'P4_backbone': p4_backbone,
            'P5_backbone': p5_backbone,
            'SPPF': sppf_feat,
            'C2PSA': c2psa_feat,
            'P4_fpn': p4_fpn,
            'P3_fpn': p3_fpn,
            'P4_pan': p4_pan,
            'P5_pan': p5_pan
        }

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

# ================== 可视化：4 行 × 10 列 ==================
def plot_all_features(img_draw, feature_dict, targets, img_size, output_path):
    device = 'cpu'
    for feat in feature_dict.values():
        if feat is not None:
            device = feat.device
            break

    target_sizes = {
        'P3_backbone': 80, 'P4_backbone': 40, 'P5_backbone': 20,
        'SPPF': 20, 'C2PSA': 20,
        'P4_fpn': 40, 'P3_fpn': 80,
        'P4_pan': 40, 'P5_pan': 20
    }

    def compute_maps(feat, size):
        if feat is None:
            return None, None, None, None
        _, _, H, W = feat.shape
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
        contrast_full = l1_map / (bg_mean + 1e-6)
        contrast_masked = contrast_full * mask
        return mean_map, l1_map, contrast_masked, contrast_full

    plt.rc('font', size=14)
    plt.rc('axes', titlesize=16)

    # 10 列：原图 + 9 特征列，原图稍宽
    fig = plt.figure(figsize=(80, 28))
    gs = gridspec.GridSpec(4, 10, figure=fig,
                           width_ratios=[2.0] + [1.0]*9,
                           wspace=0.05, hspace=0.2)

    # 原图（第 0 列，跨 4 行）
    ax_img = fig.add_subplot(gs[0:4, 0])
    ax_img.imshow(img_draw)
    ax_img.set_title('Original Image with GT', fontsize=20, fontweight='bold', pad=15)
    ax_img.axis('off')

    # 特征列顺序（按网络处理流程排列）
    col_names = ['P3_backbone', 'P4_backbone', 'P5_backbone',
                 'SPPF', 'C2PSA',
                 'P4_fpn', 'P3_fpn',
                 'P4_pan', 'P5_pan']
    row_titles = ['Mean Activation', 'L1 Norm', 'Object Contrast', 'Full Contrast']
    row_cmaps  = ['gray', 'gray', 'plasma', 'plasma']

    for col_idx, name in enumerate(col_names):
        feat = feature_dict[name]
        col_pos = col_idx + 1  # 特征列从第 1 列开始

        if feat is None:
            for row_idx in range(4):
                ax = fig.add_subplot(gs[row_idx, col_pos])
                ax.imshow(np.zeros((img_size, img_size)), cmap='gray')
                ax.set_title(f'{name}\n{row_titles[row_idx]}\n(missing)', fontsize=12)
                ax.axis('off')
            continue

        mean_map, l1_map, obj_con, full_con = compute_maps(feat, target_sizes[name])

        # 行 1：Mean
        ax1 = fig.add_subplot(gs[0, col_pos])
        ax1.imshow(normalize_and_resize(mean_map, img_size), cmap='gray')
        ax1.set_title(f'{name}\n{row_titles[0]}', fontsize=15, fontweight='bold', pad=8)
        ax1.axis('off')

        # 行 2：L1
        ax2 = fig.add_subplot(gs[1, col_pos])
        ax2.imshow(normalize_and_resize(l1_map, img_size), cmap='gray')
        ax2.set_title(f'{name}\n{row_titles[1]}', fontsize=15, fontweight='bold', pad=8)
        ax2.axis('off')

        # 行 3：Object Contrast
        ax3 = fig.add_subplot(gs[2, col_pos])
        ax3.imshow(normalize_and_resize(obj_con, img_size), cmap='plasma')
        ax3.set_title(f'{name}\n{row_titles[2]}', fontsize=15, fontweight='bold', pad=8)
        ax3.axis('off')

        # 行 4：Full Contrast
        ax4 = fig.add_subplot(gs[3, col_pos])
        ax4.imshow(normalize_and_resize(full_con, img_size), cmap='plasma')
        ax4.set_title(f'{name}\n{row_titles[3]}', fontsize=15, fontweight='bold', pad=8)
        ax4.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

# ================== 主程序 ==================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    extractor = FPNFeatureExtractor(model_path, device)

    train_img_dir = os.path.join(dataset_root, 'images', 'train')
    train_label_dir = os.path.join(dataset_root, 'labels', 'train')
    all_imgs = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.png', '.bmp'))]

    # 根据 mode 选取图片
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

        feats = extractor(img_tensor)
        print(f"Processing {img_name}: P4_pan={'OK' if feats['P4_pan'] is not None else 'MISS'}, "
              f"P5_pan={'OK' if feats['P5_pan'] is not None else 'MISS'}, ...")

        out_path = os.path.join(output_dir, f"fpn_compare_{os.path.splitext(img_name)[0]}.png")
        plot_all_features(img_draw, feats, targets, img_size, out_path)

    extractor.remove_hooks()
    print("All visualizations completed.")

if __name__ == "__main__":
    main()