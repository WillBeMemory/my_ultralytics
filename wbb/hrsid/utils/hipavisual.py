import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

from ultralytics import YOLO
from ultralytics.nn.modules import HIPA


# ================== 标签加载 ==================
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
            targets.append([cx, cy, w, h])
    return targets


def visualize_hipa_pretrained_hook(
    model_path,
    image_path,
    label_path=None,
    img_size=640,
    keep_ratio=0.25,
    save_path='hipa_pretrained_hook.png'
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---------- 1. 加载模型并移至设备 ----------
    yolo = YOLO(model_path)
    yolo.model.to(device)              # 关键修正：将整个检测模型移到 GPU
    yolo.model.eval()

    # ---------- 2. 加载图像 ----------
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_size, img_size), Image.BILINEAR)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    # ---------- 3. 加载标注框 ----------
    gt_boxes = []
    if label_path:
        targets = load_yolo_labels(label_path, img_size, img_size)
        for cx, cy, w, h in targets:
            x1 = (cx - w / 2) * img_size
            y1 = (cy - h / 2) * img_size
            x2 = (cx + w / 2) * img_size
            y2 = (cy + h / 2) * img_size
            gt_boxes.append([x1, y1, x2, y2])

    # ---------- 4. 动态查找 P5 特征层 ----------
    # 我们遍历内部 Sequential，寻找输出尺寸为 20x20 的层
    layers = yolo.model.model
    p5_layer_index = None
    # 通过一次虚拟前向（使用轻量输入）记录每层输出形状
    with torch.no_grad():
        x_test = torch.randn(1, 3, 640, 640, device=device)
        for idx, layer in enumerate(layers):
            x_test = layer(x_test)
            if isinstance(x_test, (list, tuple)):  # 如果某层输出是多个特征图（如 Detect 前的 Concat）
                continue
            if x_test.shape[-1] == 20:
                p5_layer_index = idx
                break

    if p5_layer_index is None:
        raise RuntimeError("未能找到输出尺寸为 20x20 的层，请手动指定索引。")

    print(f"选定 P5 层索引: {p5_layer_index} (输出尺寸: {x_test.shape})")

    # ---------- 5. 注册 hook 捕获 P5 特征 ----------
    p5_feature = None
    def hook_fn(module, input, output):
        nonlocal p5_feature
        p5_feature = output

    hook_handle = layers[p5_layer_index].register_forward_hook(hook_fn)

    # ---------- 6. 前向传播 ----------
    with torch.no_grad():
        _ = yolo.model(img_tensor)       # 执行整个检测流程，hook 捕获 P5 特征
    hook_handle.remove()

    if p5_feature is None:
        raise RuntimeError("hook 未捕获到特征，请检查索引。")

    print(f"捕获到的 P5 特征图形状: {p5_feature.shape}")   # 期望 (1, C, 20, 20)

    # ---------- 7. 通道适配至 256（如果必要）----------
    in_channels = p5_feature.shape[1]
    target_channels = 256
    if in_channels != target_channels:
        adapter = nn.Conv2d(in_channels, target_channels, 1, bias=False).to(device)
        p5_feat = adapter(p5_feature)
    else:
        p5_feat = p5_feature

    # ---------- 8. HIPA 稀疏化 ----------
    hipa = HIPA(
        c1=target_channels, c2=target_channels, n=1,
        num_levels=3,
        keep_ratio=keep_ratio,
        keep_ratios=[],
        min_keeps=4,
        fill_mode='center',
        use_self_attn=False,
        importance_mode='l2',
        use_gate=False
    ).to(device)
    hipa.eval()

    with torch.no_grad():
        block = hipa.blocks[0]
        residual = block.residual_proj(p5_feat)
        out_sparse, _, _, sparsity = block.hipa(p5_feat)
        final_out = residual + out_sparse

    # ---------- 9. 可视化 ----------
    def tensor_to_gray(t):
        # 取绝对值，将所有显著响应（无论正负）转为正值
        img = torch.abs(t).squeeze(0).mean(dim=0).cpu().numpy()
        vmin, vmax = img.min(), img.max()
        img = (img - vmin) / (vmax - vmin + 1e-8)
        return img

    def resize_to_input(t, size):
        img = Image.fromarray((t * 255).astype(np.uint8))
        img = img.resize((size, size), Image.NEAREST)
        return np.array(img) / 255.0

    sparse_gray = tensor_to_gray(out_sparse)
    sparse_large = resize_to_input(sparse_gray, img_size)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax0 = axes[0]
    ax0.imshow(img_np)
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                         linewidth=2, edgecolor='lime', facecolor='none')
        ax0.add_patch(rect)
    ax0.set_title('Original Image with GT')
    ax0.axis('off')

    ax1 = axes[1]
    ax1.imshow(sparse_large, cmap='gray')
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                         linewidth=2, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)
    ax1.set_title(f'Sparse Feature (keep={keep_ratio}, sparsity={sparsity.item():.4f})')
    ax1.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    model_path = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\hrsid\runs\yolo11n-lr0.01-t10\detect\train\weights\best.pt"
    image_path = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\vis\P0001_0_800_7200_8000.jpg"
    label_path = image_path.replace('.jpg', '.txt')
    if not os.path.exists(label_path):
        label_path = None

    visualize_hipa_pretrained_hook(
        model_path=model_path,
        image_path=image_path,
        label_path=label_path,
        img_size=640,
        keep_ratio=0.5,
        save_path='hipa_hook_result.png'
    )