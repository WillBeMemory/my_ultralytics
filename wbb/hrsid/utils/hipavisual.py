import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import random

# ===================== 配置参数（已使用你的路径） =====================
# 模型路径（任选一个，可自行切换）
MODEL_PATH = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\hrsid\pt\yolo11s-test-44729c-5090d.pt"
# MODEL_PATH = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\hrsid\pt\yolo11n-t10.pt"

# 数据集根目录（用于读取图片和标签）
DATASET_ROOT = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_YOLO"
TRAIN_IMG_DIR = os.path.join(DATASET_ROOT, "images", "train")
TRAIN_LABEL_DIR = os.path.join(DATASET_ROOT, "labels", "train")

# 输出目录
OUTPUT_DIR = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\hrsid\simulate\hipa_visualization"

# 图像处理参数
IMG_SIZE = 640
NUM_SAMPLES = 50                # 随机抽取几张图可视化
MODE = 'random'                # 'random' 或 'first'

# 模型层索引（P5特征图所在层，需根据你的模型结构调整）
LAYER_IDX = 5                  # yolo11s 中 model.model[5] 通常是 20x20 特征图
# 若使用模块名（优先级更高），可填写例如 "model.5"，否则留空
LAYER_NAME = ""

# HIPA 超参数
KEEP_RATIO = 0.0618
MIN_KEEPS = 0
WIN_SIZE = 3
# ===================================================================


def load_yolo_labels(label_path):
    """读取 YOLO 格式标签，返回归一化的 (cx, cy, w, h) 列表"""
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


def get_hipa_high_response(feat_map, keep_ratio=0.3, min_keeps=8, win_size=7):
    """根据 HIPA 逻辑提取高响应中心点和窗口"""
    B, C, H, W = feat_map.shape
    half = win_size // 2

    importance = torch.norm(feat_map.flatten(2), p=2, dim=1)  # (1, H*W)
    N = H * W
    k = max(min_keeps, int(N * keep_ratio))
    k = min(k, N)
    _, topk_idx = torch.topk(importance, k, dim=1)

    y_c = (topk_idx // W).long()
    x_c = (topk_idx % W).long()
    centers = torch.stack([y_c, x_c], dim=-1).squeeze(0).cpu().numpy()

    y1 = (y_c - half).clamp(0, H - 1)
    x1 = (x_c - half).clamp(0, W - 1)
    y2 = (y_c + half).clamp(0, H - 1)
    x2 = (x_c + half).clamp(0, W - 1)
    windows = torch.stack([y1, x1, y2, x2], dim=-1).squeeze(0).cpu().numpy()

    return centers, windows


def draw_hipa_and_gt(image_pil, centers, windows, gt_boxes, feat_H, feat_W, img_size):
    """在图像上绘制 HIPA 窗口/中心点 和 GT 框"""
    draw_img = image_pil.copy()
    draw = ImageDraw.Draw(draw_img)

    scale_y = img_size / feat_H
    scale_x = img_size / feat_W

    # 绘制 GT 框（绿色）
    for cx, cy, w, h in gt_boxes:
        x1 = (cx - w/2) * img_size
        y1 = (cy - h/2) * img_size
        x2 = (cx + w/2) * img_size
        y2 = (cy + h/2) * img_size
        draw.rectangle([x1, y1, x2, y2], outline='lime', width=2)

    # 绘制窗口（蓝色）
    for y1, x1, y2, x2 in windows:
        rect = [x1 * scale_x, y1 * scale_y, (x2 + 1) * scale_x, (y2 + 1) * scale_y]
        draw.rectangle(rect, outline='blue', width=2)

    # 绘制中心点（红色实心圆）
    for y, x in centers:
        cx = x * scale_x
        cy = y * scale_y
        r = 4
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill='red', outline='red')

    return draw_img


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 加载模型
    yolo = YOLO(MODEL_PATH, verbose=False)
    model = yolo.model
    model.eval()
    model.to(device)

    # 定位目标层
    target_layer = None
    if LAYER_NAME:
        for name, module in model.named_modules():
            if name == LAYER_NAME:
                target_layer = module
                break
        if target_layer is None:
            raise ValueError(f"Layer '{LAYER_NAME}' not found.")
    else:
        try:
            target_layer = model.model[LAYER_IDX]
        except IndexError:
            raise ValueError(f"Layer index {LAYER_IDX} out of range. model.model has {len(model.model)} layers.")

    feat_container = []

    def hook_fn(module, input, output):
        if isinstance(output, (list, tuple)):
            feat = output[0].detach()
        else:
            feat = output.detach()
        if len(feat.shape) == 4:
            feat_container.append(feat)

    hook = target_layer.register_forward_hook(hook_fn)

    # 收集待处理图像
    all_imgs = [f for f in os.listdir(TRAIN_IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    if MODE == 'first':
        selected_imgs = all_imgs[:min(NUM_SAMPLES, len(all_imgs))]
    elif MODE == 'random':
        selected_imgs = random.sample(all_imgs, min(NUM_SAMPLES, len(all_imgs)))
    else:
        raise ValueError(f"Unknown mode: {MODE}. Use 'random' or 'first'.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for img_name in selected_imgs:
        img_path = os.path.join(TRAIN_IMG_DIR, img_name)
        label_path = os.path.join(TRAIN_LABEL_DIR, os.path.splitext(img_name)[0] + '.txt')

        print(f"Processing: {img_path}")
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img_resized) / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)

        # 加载 GT 框
        gt_boxes = load_yolo_labels(label_path)

        # 前向推理，捕获特征图
        feat_container.clear()
        with torch.no_grad():
            _ = model(img_tensor)

        if not feat_container:
            print(f"  Warning: no feature map captured. Check layer index/name.")
            continue

        p5_feat = feat_container[0]
        _, C, H, W = p5_feat.shape
        print(f"  Feature shape: {p5_feat.shape}")

        centers, windows = get_hipa_high_response(
            p5_feat,
            keep_ratio=KEEP_RATIO,
            min_keeps=MIN_KEEPS,
            win_size=WIN_SIZE
        )
        print(f"  Selected {len(centers)} high‑response centers.")

        result_img = draw_hipa_and_gt(img_resized, centers, windows, gt_boxes, H, W, IMG_SIZE)
        base_name = os.path.splitext(img_name)[0]
        save_path = os.path.join(OUTPUT_DIR, f"{base_name}_hipa.png")
        result_img.save(save_path)
        print(f"  Saved to {save_path}")

    hook.remove()
    print("All done.")


if __name__ == "__main__":
    main()