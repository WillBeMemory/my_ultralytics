import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import random

from ultralytics import YOLO
from ultralytics.nn.modules import SPDConv


# ================== 导入 SPDConv 模块 ==================
# 请确保 spd_conv.py 在当前目录或 Python 路径中，其中定义了 SPDConv



# ================== 固定 Backbone（提取 P3） ==================
class PretrainedBackboneP3(nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        yolo = YOLO(model_path, verbose=False)
        self.model = yolo.model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.device = device
        self.model.to(device)

        self.p3_feature = None
        self._register_hook()

    def _register_hook(self):
        def hook_fn(module, input, output):
            features = output
            if isinstance(features, (list, tuple)):
                for feat in features:
                    if isinstance(feat, torch.Tensor) and feat.shape[-1] == 80:
                        self.p3_feature = feat
                        break
            elif isinstance(features, torch.Tensor) and features.shape[-1] == 80:
                self.p3_feature = features

        self.hook_handles = []
        for layer in self.model.model:
            self.hook_handles.append(layer.register_forward_hook(hook_fn))

    def forward(self, x):
        with torch.no_grad():
            self.p3_feature = None
            _ = self.model(x)
            if self.p3_feature is None:
                raise RuntimeError("无法捕获 P3 特征")
            return self.p3_feature

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()


# ================== 简易检测头（适配 40×40） ==================
class SimpleDetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 5, 1)   # obj, tx, ty, tw, th
        )
        nn.init.constant_(self.conv[-1].bias[0], -4.6)

    def forward(self, x):
        return self.conv(x)


# ================== 检测损失（特征图尺寸 40×40） ==================
def detection_loss(pred, targets, feat_h=40, feat_w=40):
    B = pred.shape[0]
    obj_pred = pred[:, 0:1, :, :]
    tx = pred[:, 1:2, :, :]
    ty = pred[:, 2:3, :, :]
    tw = pred[:, 3:4, :, :]
    th = pred[:, 4:5, :, :]

    obj_target = torch.zeros_like(obj_pred)
    tx_target = torch.zeros_like(tx)
    ty_target = torch.zeros_like(ty)
    tw_target = torch.zeros_like(tw)
    th_target = torch.zeros_like(th)

    for b in range(B):
        for cx, cy, w, h in targets[b]:
            cx_f = cx * feat_w
            cy_f = cy * feat_h
            w_f = w * feat_w
            h_f = h * feat_h
            gx = int(cx_f)
            gy = int(cy_f)
            if not (0 <= gx < feat_w and 0 <= gy < feat_h):
                continue
            obj_target[b, 0, gy, gx] = 1.0
            tx_target[b, 0, gy, gx] = cx_f - gx
            ty_target[b, 0, gy, gx] = cy_f - gy
            tw_target[b, 0, gy, gx] = torch.log(torch.tensor(w_f, device=pred.device) + 1e-6)
            th_target[b, 0, gy, gx] = torch.log(torch.tensor(h_f, device=pred.device) + 1e-6)

    pos_weight = torch.tensor([10.0], device=pred.device)
    loss_obj = F.binary_cross_entropy_with_logits(obj_pred, obj_target, pos_weight=pos_weight)
    mask = obj_target > 0.5
    loss_box = 0.0
    if mask.sum() > 0:
        loss_box += F.mse_loss(tx[mask], tx_target[mask])
        loss_box += F.mse_loss(ty[mask], ty_target[mask])
        loss_box += F.mse_loss(tw[mask], tw_target[mask])
        loss_box += F.mse_loss(th[mask], th_target[mask])
    return loss_obj + loss_box


# ================== 可视化辅助函数 ==================
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

def compute_maps(feat, targets, img_size):
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


def visualize_results(img_draw, p3_raw, p3_spd, targets, img_size, save_path):
    """
    生成 4 行 2 列的对比图
    行: Mean, L1, Object Contrast, Full Contrast
    列: P3 Backbone, P3 + SPDConv
    """
    maps_raw = compute_maps(p3_raw.cpu(), targets, img_size)
    maps_spd = compute_maps(p3_spd.cpu(), targets, img_size)

    plt.rc('font', size=14)
    plt.rc('axes', titlesize=16)
    fig = plt.figure(figsize=(18, 24))
    gs = gridspec.GridSpec(4, 2, figure=fig,
                           width_ratios=[1.5, 1.5],
                           wspace=0.05, hspace=0.25)

    row_titles = ['Mean Activation', 'L1 Norm', 'Object Contrast', 'Full Contrast']
    row_cmaps  = ['gray', 'gray', 'plasma', 'plasma']
    col_titles = ['P3 Backbone (80×80)', 'SPDConv Output (40×40)']

    for row_idx in range(4):
        for col_idx, (maps, title) in enumerate(zip([maps_raw, maps_spd], col_titles)):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            feat_map = maps[row_idx]
            ax.imshow(normalize_and_resize(feat_map, img_size), cmap=row_cmaps[row_idx])
            ax.set_title(title, fontsize=15, fontweight='bold', pad=8)
            ax.axis('off')

    # 在顶部添加原图（由于空间有限，可将原图放入一个独立的行或列，这里省略，原图可在单独子图中显示）
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


# ================== 主实验 ==================
def run_experiment(model_path, dataset_root, output_dir,
                   num_samples=50, epochs=200, batch_size=8, lr=0.002):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(output_dir, exist_ok=True)

    # ---------- 准备数据 ----------
    train_img_dir = os.path.join(dataset_root, 'images', 'train')
    train_label_dir = os.path.join(dataset_root, 'labels', 'train')
    all_imgs = [f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.png'))]
    selected_imgs = random.sample(all_imgs, min(num_samples, len(all_imgs)))

    data = []
    for img_name in selected_imgs:
        img_path = os.path.join(train_img_dir, img_name)
        label_path = os.path.join(train_label_dir, os.path.splitext(img_name)[0] + '.txt')

        img = Image.open(img_path).convert('RGB')
        img = img.resize((640, 640), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        targets = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    cls, cx, cy, w, h = map(float, parts)
                    targets.append([cx, cy, w, h])

        data.append((img_tensor, targets, img_name))

    # ---------- 加载 Backbone ----------
    backbone = PretrainedBackboneP3(model_path, device)
    dummy_img = data[0][0].unsqueeze(0).to(device)
    with torch.no_grad():
        p3_sample = backbone(dummy_img)
        in_channels = p3_sample.shape[1]
    print(f"P3 channels: {in_channels}")

    # ---------- 实例化 SPDConv（保持通道数不变，纯下采样到 40×40） ----------
    spd_conv = SPDConv(c1=in_channels, c2=in_channels, kernel_size=3).to(device)
    head = SimpleDetectionHead(in_channels).to(device)

    optimizer = optim.SGD(
        list(spd_conv.parameters()) + list(head.parameters()),
        lr=lr, momentum=0.9, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)

    # ---------- 训练 ----------
    spd_conv.train()
    head.train()
    for epoch in range(epochs):
        random.shuffle(data)
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            imgs = torch.stack([d[0] for d in batch_data]).to(device)
            targets_batch = [d[1] for d in batch_data]

            optimizer.zero_grad()
            with torch.no_grad():
                p3_feats = backbone(imgs)
            enhanced = spd_conv(p3_feats)          # 输出 (B, C, 40, 40)
            pred = head(enhanced)
            loss = detection_loss(pred, targets_batch)  # feat_h=feat_w=40
            if torch.isnan(loss):
                print(f"NaN at epoch {epoch+1}")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(spd_conv.parameters()) + list(head.parameters()), 10.0
            )
            optimizer.step()
        scheduler.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    print("训练完成！")

    # ---------- 可视化 ----------
    spd_conv.eval()
    head.eval()
    for img_tensor, targets, img_name in data:
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_t = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            p3_raw = backbone(img_t)
            p3_spd = spd_conv(p3_raw)

        # 原图 + 标注
        img_draw = Image.fromarray((img_np * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img_draw)
        for cx, cy, w, h in targets:
            x1 = (cx - w/2) * 640
            y1 = (cy - h/2) * 640
            x2 = (cx + w/2) * 640
            y2 = (cy + h/2) * 640
            draw.rectangle([x1, y1, x2, y2], outline='lime', width=2)

        out_path = os.path.join(output_dir, f"vis_{img_name}.png")
        visualize_results(img_draw, p3_raw, p3_spd, targets, 640, out_path)

    backbone.remove_hooks()
    print("All visualizations saved to", output_dir)


# ================== 入口 ==================
if __name__ == "__main__":
    model_path = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\hrsid\pt\hrsid-yolov8n-5090d.pt"
    dataset_root = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_YOLO"
    output_dir = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\hrsid\simulate\spdconv_p3_visual"

    run_experiment(
        model_path=model_path,
        dataset_root=dataset_root,
        output_dir=output_dir,
        num_samples=50,
        epochs=200,
        batch_size=8,
        lr=0.002
    )