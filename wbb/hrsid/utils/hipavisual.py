import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import os

# ===================== 配置 =====================
MODEL_PATH = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\hrsid\pt\yolo11s-test-44729c-5090d.pt"
DATASET_ROOT = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_COMPLEX"
TRAIN_IMG_DIR = os.path.join(DATASET_ROOT, "images", "train")
TRAIN_LABEL_DIR = os.path.join(DATASET_ROOT, "labels", "train")
OUTPUT_DIR = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\hrsid\simulate\hipa_level_vis"

IMG_SIZE = 640
MAX_IMAGES = 200
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LAYER_IDX = 5

KEEP_RATIO = 0.3
MIN_KEEPS = 4
LOCAL_RANGE_KERNEL = 3

# 允许的窗口尺寸（从大到小）
ALLOWED_SIZES = [7, 5, 3, 1]
MIN_POINTS_IN_WINDOW = 2   # 合并时窗口内至少需要多少个未覆盖点

SIZE_COLORS = {
    1: (255, 0, 0),    # 红
    3: (0, 255, 0),    # 绿
    5: (0, 0, 255),    # 蓝
    7: (255, 255, 0),  # 黄
}
GT_COLOR = (255, 255, 255)       # 白色
# =================================================


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


def compute_local_range(feat, kernel_size=3):
    max_val = F.max_pool2d(feat, kernel_size, stride=1, padding=kernel_size // 2)
    min_val = -F.max_pool2d(-feat, kernel_size, stride=1, padding=kernel_size // 2)
    return (max_val - min_val).mean(dim=1)


def get_initial_points(feat_map, keep_ratio, min_keeps):
    B, C, H, W = feat_map.shape
    N = H * W
    range_map = compute_local_range(feat_map, LOCAL_RANGE_KERNEL)
    flat_range = range_map.flatten(1)
    k = max(min_keeps, int(N * keep_ratio))
    k = min(k, N)
    topk_idx = torch.topk(flat_range, k=k, dim=-1)[1].squeeze(0)
    ys = (topk_idx // W).cpu().numpy()
    xs = (topk_idx % W).cpu().numpy()
    return np.stack([ys, xs], axis=-1), H, W


def greedy_merge_non_overlap(points, H, W, allowed_sizes, min_pts=2):
    """
    贪婪放置窗口：每次选择一个窗口（尺寸从大到小），要求窗口不重叠且覆盖至少 min_pts 个未覆盖点。
    剩余未覆盖点保留为 1x1。
    返回 [(y1, x1, y2, x2, size), ...]
    """
    # 标记未覆盖的选中点
    uncovered = set((int(y), int(x)) for y, x in points)
    covered_mask = np.zeros((H, W), dtype=bool)
    windows = []

    # 对每个尺寸（除1x1）进行贪婪放置
    for size in allowed_sizes:
        if size == 1:
            continue
        half = size // 2
        while True:
            best_win = None
            best_count = 0
            # 遍历所有未覆盖点作为可能的窗口中心
            for (y, x) in list(uncovered):
                y1 = max(0, y - half)
                x1 = max(0, x - half)
                y2 = min(H - 1, y + half)
                x2 = min(W - 1, x + half)

                # 检查是否与已覆盖区域重叠
                if np.any(covered_mask[y1:y2+1, x1:x2+1]):
                    continue

                # 统计窗口内未覆盖的选中点数量
                cnt = 0
                for dy in range(y1, y2 + 1):
                    for dx in range(x1, x2 + 1):
                        if (dy, dx) in uncovered:
                            cnt += 1
                if cnt >= min_pts and cnt > best_count:
                    best_count = cnt
                    best_win = (y1, x1, y2, x2)

            if best_win is None:
                break   # 该尺寸无法再放置

            y1, x1, y2, x2 = best_win
            # 移除窗口内所有未覆盖点
            for dy in range(y1, y2 + 1):
                for dx in range(x1, x2 + 1):
                    uncovered.discard((dy, dx))
            # 标记覆盖区域
            covered_mask[y1:y2+1, x1:x2+1] = True
            windows.append((y1, x1, y2, x2, size))

    # 剩余未覆盖点作为 1x1 窗口
    for (y, x) in uncovered:
        # 如果该点未被覆盖掩码标记，直接添加（理论上 uncovered 和 covered_mask 应该一致，但安全起见检查）
        if not covered_mask[y, x]:
            covered_mask[y, x] = True
            windows.append((y, x, y, x, 1))

    return windows


def draw_windows_and_gt(image_pil, windows, gt_boxes, img_size, feat_h, feat_w):
    draw_img = image_pil.copy()
    draw = ImageDraw.Draw(draw_img)
    scale_y = img_size / feat_h
    scale_x = img_size / feat_w

    # GT 白色
    for cx, cy, w, h in gt_boxes:
        x1 = (cx - w/2) * img_size
        y1 = (cy - h/2) * img_size
        x2 = (cx + w/2) * img_size
        y2 = (cy + h/2) * img_size
        draw.rectangle([x1, y1, x2, y2], outline=GT_COLOR, width=3)

    # 窗口
    for y1, x1, y2, x2, size in windows:
        color = SIZE_COLORS.get(size, (128,128,128))
        img_x1 = x1 * scale_x
        img_y1 = y1 * scale_y
        img_x2 = (x2 + 1) * scale_x
        img_y2 = (y2 + 1) * scale_y
        draw.rectangle([img_x1, img_y1, img_x2, img_y2], outline=color, width=2)
    return draw_img


def main():
    print(f"Using device: {DEVICE}")
    yolo = YOLO(MODEL_PATH, verbose=False)
    model = yolo.model.eval().to(DEVICE)
    target_layer = model.model[LAYER_IDX]
    feat_container = []
    def hook_fn(m, i, o):
        f = o[0] if isinstance(o, (list, tuple)) else o
        if len(f.shape) == 4:
            feat_container.append(f.detach())
    hook = target_layer.register_forward_hook(hook_fn)

    all_imgs = sorted([f for f in os.listdir(TRAIN_IMG_DIR) if f.lower().endswith(('.jpg','.png','.jpeg','.bmp'))])
    selected_imgs = all_imgs[:min(MAX_IMAGES, len(all_imgs))]
    print(f"Processing {len(selected_imgs)} images")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for img_name in selected_imgs:
        img_path = os.path.join(TRAIN_IMG_DIR, img_name)
        label_path = os.path.join(TRAIN_LABEL_DIR, os.path.splitext(img_name)[0] + '.txt')
        img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        tensor = torch.from_numpy(np.array(img)/255.0).permute(2,0,1).unsqueeze(0).float().to(DEVICE)
        gt_boxes = load_yolo_labels(label_path)

        feat_container.clear()
        with torch.no_grad():
            _ = model(tensor)
        if not feat_container:
            continue
        p5_feat = feat_container[0]
        H, W = p5_feat.shape[2:]

        points, _, _ = get_initial_points(p5_feat, KEEP_RATIO, MIN_KEEPS)
        windows = greedy_merge_non_overlap(points, H, W, ALLOWED_SIZES, MIN_POINTS_IN_WINDOW)

        result = draw_windows_and_gt(img, windows, gt_boxes, IMG_SIZE, H, W)
        save_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_name)[0]}_greedy_nomerge_loss.png")
        result.save(save_path)

    hook.remove()
    print("Done.")


if __name__ == "__main__":
    main()