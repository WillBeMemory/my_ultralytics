import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import os

# ===================== 配置参数 =====================
MODEL_PATH = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\hrsid\pt\yolo11s-test-44729c-5090d.pt"
DATASET_ROOT = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_COMPLEX"
TRAIN_IMG_DIR = os.path.join(DATASET_ROOT, "images", "train")
TRAIN_LABEL_DIR = os.path.join(DATASET_ROOT, "labels", "train")
OUTPUT_DIR = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\hrsid\simulate\hipa_level_vis"

IMG_SIZE = 640
MAX_IMAGES = 200                    # 处理前200张图片
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LAYER_IDX = 5                       # P5 特征图所在层

# top‑k 保留参数
KEEP_RATIO = 0.3                    # 保留比例
MIN_KEEPS = 4                       # 最少保留网格数

# 局部极差窗口大小
LOCAL_RANGE_KERNEL = 3              # 窗口大小（通常为奇数）

# 可视化颜色
HIPA_COLOR = (255, 0, 0)            # 选中网格框（红色）
GT_COLOR = (0, 255, 0)              # 真实目标框（亮绿）
# ===================================================


def load_yolo_labels(label_path):
    """读取 YOLO 格式标签，返回归一化 (cx, cy, w, h) 列表"""
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


def compute_local_range(feat, kernel_size=3):
    """
    计算特征图每个空间位置的局部极差（max - min），沿通道取平均。
    feat: (1, C, H, W)
    返回: (1, H, W) 局部极差图
    """
    # 局部最大值池化
    max_val = F.max_pool2d(feat, kernel_size, stride=1, padding=kernel_size // 2)
    # 局部最小值池化（用负输入的最大池化技巧）
    min_val = -F.max_pool2d(-feat, kernel_size, stride=1, padding=kernel_size // 2)
    # 极差，沿通道平均得到单通道响应图
    range_val = (max_val - min_val).mean(dim=1)   # (1, H, W)
    return range_val


def select_topk_from_map(scores, keep_ratio, min_keeps, H, W):
    """
    从分数图 (1, H, W) 中选取 top‑k 网格，返回归一化坐标 (1, K, 4)
    """
    N = H * W
    scores_flat = scores.flatten(1)                         # (1, N)

    k = max(min_keeps, int(N * keep_ratio))
    k = min(k, N)
    topk_idx = torch.topk(scores_flat, k=k, dim=-1)[1]      # (1, k)

    y_idx = topk_idx // W   # (1, k)
    x_idx = topk_idx % W    # (1, k)

    grid_h = 1.0 / H
    grid_w = 1.0 / W
    cx = (x_idx.float() + 0.5) * grid_w
    cy = (y_idx.float() + 0.5) * grid_h
    w = torch.full_like(cx, grid_w)
    h = torch.full_like(cy, grid_h)

    boxes = torch.stack([cx, cy, w, h], dim=-1)   # (1, k, 4)
    return boxes


def draw_results(image_pil, selected_boxes, gt_boxes, img_size):
    """绘制 GT 框（绿色）和局部极差高响应框（红色）"""
    draw_img = image_pil.copy()
    draw = ImageDraw.Draw(draw_img)
    W_img, H_img = img_size, img_size

    for cx, cy, w, h in gt_boxes:
        x1 = (cx - w/2) * W_img
        y1 = (cy - h/2) * H_img
        x2 = (cx + w/2) * W_img
        y2 = (cy + h/2) * H_img
        draw.rectangle([x1, y1, x2, y2], outline=GT_COLOR, width=3)

    if selected_boxes.shape[1] > 0:
        boxes_np = selected_boxes.squeeze(0).cpu().numpy()
        for bbox in boxes_np:
            cx, cy, w, h = bbox
            x1 = (cx - w/2) * W_img
            y1 = (cy - h/2) * H_img
            x2 = (cx + w/2) * W_img
            y2 = (cy + h/2) * H_img
            draw.rectangle([x1, y1, x2, y2], outline=HIPA_COLOR, width=2)
    return draw_img


def main():
    print(f"Using device: {DEVICE}")

    yolo = YOLO(MODEL_PATH, verbose=False)
    model = yolo.model
    model.eval()
    model.to(DEVICE)

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

    all_imgs = sorted([f for f in os.listdir(TRAIN_IMG_DIR)
                       if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))])
    selected_imgs = all_imgs[:min(MAX_IMAGES, len(all_imgs))]
    print(f"Total images: {len(all_imgs)}, processing first {len(selected_imgs)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for img_name in selected_imgs:
        img_path = os.path.join(TRAIN_IMG_DIR, img_name)
        label_path = os.path.join(TRAIN_LABEL_DIR, os.path.splitext(img_name)[0] + '.txt')
        print(f"Processing {img_path} ...")

        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img_resized)/255.0).permute(2,0,1).unsqueeze(0).float().to(DEVICE)

        gt_boxes = load_yolo_labels(label_path)

        feat_container.clear()
        with torch.no_grad():
            _ = model(img_tensor)

        if not feat_container:
            print(f"  未捕获到特征图，跳过 {img_name}")
            continue

        p5_feat = feat_container[0]   # (1, C, H, W)
        H, W = p5_feat.shape[2:]
        print(f"  P5 特征图形状: {p5_feat.shape}")

        # 计算局部极差
        range_map = compute_local_range(p5_feat, kernel_size=LOCAL_RANGE_KERNEL)

        # 选取 top‑k 网格
        selected_boxes = select_topk_from_map(range_map, KEEP_RATIO, MIN_KEEPS, H, W)

        k = selected_boxes.shape[1]
        print(f"  局部极差 top‑k 网格数: {k}，GT 框数: {len(gt_boxes)}")

        result_img = draw_results(img_resized, selected_boxes, gt_boxes, IMG_SIZE)
        base_name = os.path.splitext(img_name)[0]
        save_path = os.path.join(OUTPUT_DIR, f"{base_name}_localrange_topk.png")
        result_img.save(save_path)
        print(f"  保存至 {save_path}")

    hook.remove()
    print("全部完成。")


if __name__ == "__main__":
    main()