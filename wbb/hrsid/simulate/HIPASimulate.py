import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import random
from torch.utils.data import Dataset, DataLoader

from ultralytics import YOLO
from ultralytics.nn.modules import HIPA   # 确保已导入最新版 HIPA（含 aggregate）


# ================== 数据集 ==================
class HRSIDDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640, augment=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment
        self.img_list = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')

        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
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
        return img_tensor, targets, img_name


# ================== 固定 backbone（提取 P5） ==================
class PretrainedBackbone(nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        yolo = YOLO(model_path)
        self.model = yolo.model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.device = device
        self.model.to(device)

        self.p5_feature = None
        self._register_p5_hook()

    def _register_p5_hook(self):
        def hook_fn(module, input, output):
            features = output
            if isinstance(features, (list, tuple)):
                for feat in features:
                    if isinstance(feat, torch.Tensor) and feat.shape[-1] == 20:
                        self.p5_feature = feat
                        break
            elif isinstance(features, torch.Tensor) and features.shape[-1] == 20:
                self.p5_feature = features

        self.hook_handles = []
        for layer in self.model.model:
            self.hook_handles.append(layer.register_forward_hook(hook_fn))

    def forward(self, x):
        with torch.no_grad():
            self.p5_feature = None
            _ = self.model(x)
            if self.p5_feature is None:
                raise RuntimeError("无法捕获 P5 特征")
            return self.p5_feature

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()


# ================== HIPA + 检测头 ==================
class HIPAHead(nn.Module):
    def __init__(self, in_channels, hipa_channels=256, num_levels=3,
                 threshold_init=0.5, fill_mode='constant',
                 aggregate=False, use_self_attn=False, importance_mode='l2'):
        super().__init__()
        if in_channels != hipa_channels:
            self.align_conv = nn.Conv2d(in_channels, hipa_channels, 1, bias=False)
        else:
            self.align_conv = nn.Identity()

        self.hipa = HIPA(
            c1=hipa_channels, c2=hipa_channels, n=1,
            num_levels=num_levels,
            threshold_init=threshold_init,
            use_contrast_norm=True,
            use_self_attn=use_self_attn,
            importance_mode=importance_mode,
            fill_mode=fill_mode,
            aggregate=aggregate,
        )
        self.head = nn.Sequential(
            nn.Conv2d(hipa_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 5, 1)   # obj, tx, ty, tw, th
        )
        nn.init.constant_(self.head[-1].bias[0], -4.6)

    def forward(self, x):
        x = self.align_conv(x)
        x = self.hipa(x, None)   # 依赖注入的 target_mask 属性
        return self.head(x)

    def get_sparse_output(self, x):
        with torch.no_grad():
            x = self.align_conv(x)
            out_sparse = self.hipa(x, None)
        return out_sparse


# ================== 损失函数（仅检测损失） ==================
def compute_loss(pred, targets, feat_h=20, feat_w=20):
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
            cx_f = cx * feat_w; cy_f = cy * feat_h
            w_f = w * feat_w; h_f = h * feat_h
            gx = int(cx_f); gy = int(cy_f)
            if not (0 <= gx < feat_w and 0 <= gy < feat_h): continue
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


# ================== 训练与可视化 ==================
def run_experiment(
    model_path, dataset_root, output_dir,
    num_samples=50, epochs=200, batch_size=8, lr=0.002,
    threshold_init=0.5, fill_mode='constant', aggregate=False,
    use_self_attn=False,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(output_dir, exist_ok=True)

    # 1. 随机选取图像
    train_img_dir = os.path.join(dataset_root, 'images', 'train')
    train_label_dir = os.path.join(dataset_root, 'labels', 'train')
    all_imgs = [f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.png'))]
    selected_imgs = random.sample(all_imgs, min(num_samples, len(all_imgs)))
    img_paths = [os.path.join(train_img_dir, f) for f in selected_imgs]
    label_paths = [os.path.join(train_label_dir, os.path.splitext(f)[0] + '.txt') for f in selected_imgs]

    data = []
    for img_path, label_path in zip(img_paths, label_paths):
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
        data.append((img_tensor, targets, os.path.basename(img_path)))

    backbone = PretrainedBackbone(model_path, device)

    dummy_img = data[0][0].unsqueeze(0).to(device)
    with torch.no_grad():
        p5_sample = backbone(dummy_img)
        in_channels = p5_sample.shape[1]

    hipa_head = HIPAHead(
        in_channels, hipa_channels=256, num_levels=3,
        threshold_init=threshold_init, fill_mode=fill_mode,
        aggregate=aggregate, use_self_attn=use_self_attn,
    ).to(device)

    optimizer = optim.SGD(hipa_head.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)

    # 训练循环
    hipa_head.train()
    for epoch in range(epochs):
        random.shuffle(data)
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            imgs = torch.stack([d[0] for d in batch_data]).to(device)
            targets_batch = [d[1] for d in batch_data]

            # 生成 P5 目标掩膜 (20x20)
            B = len(imgs)
            target_mask = torch.zeros(B, 20, 20, device=device)
            for b, tgts in enumerate(targets_batch):
                for cx, cy, w, h in tgts:
                    cx_f = cx * 20; cy_f = cy * 20
                    w_f = w * 20; h_f = h * 20
                    x1 = int(max(0, cx_f - w_f/2))
                    y1 = int(max(0, cy_f - h_f/2))
                    x2 = int(min(20, cx_f + w_f/2 + 1))
                    y2 = int(min(20, cy_f + h_f/2 + 1))
                    if x2 > x1 and y2 > y1:
                        target_mask[b, y1:y2, x1:x2] = 1.0

            # 注入 target_mask 到 HIPABlock
            hipa_head.hipa.hipa.target_mask = target_mask

            optimizer.zero_grad()
            with torch.no_grad():
                p5_feats = backbone(imgs)
            pred = hipa_head(p5_feats)
            loss = compute_loss(pred, targets_batch)

            # 清除注入的掩膜
            hipa_head.hipa.hipa.target_mask = None

            if torch.isnan(loss):
                print(f"NaN at epoch {epoch+1}")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hipa_head.parameters(), 10.0)
            optimizer.step()
        scheduler.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    print("训练完成！")

    # 可视化（推理时不传 target_mask）
    hipa_head.eval()
    for img_tensor, targets, img_name in data:
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_t = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            p5 = backbone(img_t)
            sparse = hipa_head.get_sparse_output(p5)

        p5_feat = p5.squeeze(0).abs().mean(dim=0).cpu().numpy()
        p5_feat = (p5_feat - p5_feat.min()) / (p5_feat.max() - p5_feat.min() + 1e-8)
        p5_img = Image.fromarray((p5_feat * 255).astype(np.uint8)).resize((640, 640), Image.NEAREST)
        p5_large = np.array(p5_img) / 255.0

        sparse_abs = sparse.squeeze(0).abs().mean(dim=0).cpu().numpy()
        sparse_abs = (sparse_abs - sparse_abs.min()) / (sparse_abs.max() - sparse_abs.min() + 1e-8)
        sparse_pil = Image.fromarray((sparse_abs * 255).astype(np.uint8)).resize((640, 640), Image.NEAREST)
        sparse_large = np.array(sparse_pil) / 255.0

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(img_np)
        for cx, cy, w, h in targets:
            x1 = (cx - w/2) * 640; y1 = (cy - h/2) * 640
            x2 = (cx + w/2) * 640; y2 = (cy + h/2) * 640
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='lime', facecolor='none')
            axes[0].add_patch(rect)
        axes[0].set_title('Original Image with GT')
        axes[0].axis('off')

        axes[1].imshow(p5_large, cmap='gray')
        axes[1].set_title('P5 Feature Map (20x20)')
        axes[1].axis('off')

        axes[2].imshow(sparse_large, cmap='gray')
        for cx, cy, w, h in targets:
            x1 = (cx - w/2) * 640; y1 = (cy - h/2) * 640
            x2 = (cx + w/2) * 640; y2 = (cy + h/2) * 640
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
            axes[2].add_patch(rect)
        axes[2].set_title('Sparse Feature (HIPA) with GT')
        axes[2].axis('off')

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"vis_{img_name}.png")
        plt.savefig(out_path, dpi=100)
        plt.close()
        print(f"Saved {out_path}")

    print("All visualizations saved to", output_dir)
    backbone.remove_hooks()


# ================== 主程序 ==================
if __name__ == "__main__":
    model_path = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\hrsid\runs\yolo11n-lr0.01-t10\detect\train\weights\best.pt"
    dataset_root = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_YOLO"
    output_dir = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\hrsid\simulate\hipa_visual_output"

    run_experiment(
        model_path=model_path,
        dataset_root=dataset_root,
        output_dir=output_dir,
        num_samples=50,
        epochs=200,
        batch_size=8,
        lr=0.002,
        threshold_init=0.5,
        fill_mode='upsample',      # 可改为 'upsample'
        aggregate=True,           # 可改为 True 以启用多级聚合
        use_self_attn=False,
    )