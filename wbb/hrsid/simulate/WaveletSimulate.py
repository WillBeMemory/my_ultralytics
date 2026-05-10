import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from ultralytics.nn.modules import WaveletStem


# ================== 简单检测头 ==================
class SimpleDetectionHead(nn.Module):
    """极简检测头，输出 [obj, tx, ty, tw, th]"""
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 5, 1)   # 5 通道: obj, tx, ty, tw, th
        )

    def forward(self, x):
        return self.conv(x)   # (B,5,H,W)


# ================== 损失函数 ==================
def detection_loss(pred, targets, stride=2):
    """
    pred: (1, 5, H, W)  -> [obj, tx, ty, tw, th]
    targets: list of [cx, cy, w, h] 归一化坐标
    """
    B, C, Hf, Wf = pred.shape
    obj_pred = pred[:, 0:1, :, :]   # (1,1,Hf,Wf)
    tx = pred[:, 1:2, :, :]
    ty = pred[:, 2:3, :, :]
    tw = pred[:, 3:4, :, :]
    th = pred[:, 4:5, :, :]

    obj_target = torch.zeros_like(obj_pred)
    tx_target = torch.zeros_like(tx)
    ty_target = torch.zeros_like(ty)
    tw_target = torch.zeros_like(tw)
    th_target = torch.zeros_like(th)

    if len(targets) == 0:
        bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
        return bce_loss(obj_pred, obj_target)

    for target in targets:
        cx, cy, w, h = target
        cx_f = cx * Wf
        cy_f = cy * Hf
        w_f = w * Wf
        h_f = h * Hf

        gx = int(cx_f)
        gy = int(cy_f)
        if not (0 <= gx < Wf and 0 <= gy < Hf):
            continue

        obj_target[0, 0, gy, gx] = 1.0
        tx_target[0, 0, gy, gx] = cx_f - gx
        ty_target[0, 0, gy, gx] = cy_f - gy
        tw_target[0, 0, gy, gx] = torch.log(torch.tensor(w_f, device=pred.device) + 1e-6)
        th_target[0, 0, gy, gx] = torch.log(torch.tensor(h_f, device=pred.device) + 1e-6)

    bce = nn.BCEWithLogitsLoss(reduction='sum')
    loss_obj = bce(obj_pred, obj_target)

    mask = obj_target > 0.5
    loss_box = 0.0
    if mask.sum() > 0:
        loss_box += F.mse_loss(tx[mask], tx_target[mask], reduction='sum')
        loss_box += F.mse_loss(ty[mask], ty_target[mask], reduction='sum')
        loss_box += F.mse_loss(tw[mask], tw_target[mask], reduction='sum')
        loss_box += F.mse_loss(th[mask], th_target[mask], reduction='sum')

    return loss_obj + loss_box


# ================== 读取 YOLO 标注 ==================
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


# ================== 主程序 ==================
def main():
    # 配置
    image_path = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\vis\P0001_0_800_7200_8000.jpg"
    label_path = image_path.replace('.jpg', '.txt')
    in_channels = 3
    out_channels = 64
    input_size = 640
    lr = 0.005                       # 降低初始学习率，避免 NaN
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 100
    conf_thresh = 0.1                # 显示阈值设低一些，确保有预测框
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载图像
    img = Image.open(image_path).convert('RGB')
    img = img.resize((input_size, input_size), Image.BILINEAR)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,640,640)

    # 加载标注
    targets = load_yolo_labels(label_path, img.width, img.height)

    # 构建模型
    stem = WaveletStem(in_channels=in_channels, out_channels=out_channels, use_diag_init=True).to(device)
    head = SimpleDetectionHead(out_channels).to(device)
    model = nn.Sequential(stem, head)
    model.train()

    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.001)

    # 训练循环
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(img_tensor)               # (1,5,320,320)
        loss = detection_loss(pred, targets, stride=2)
        if torch.isnan(loss):
            print(f"Loss is NaN at epoch {epoch+1}. Stopping.")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        if (epoch + 1) % 30 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}, Loss: {loss.item():.4f}")

    print("训练完成！")

    # 评估模式并提取特征
    model.eval()
    with torch.no_grad():
        feat = stem(img_tensor)                # (1,64,320,320)
        pred = model(img_tensor)               # (1,5,320,320)

    # 解码预测框
    Hf, Wf = pred.shape[2], pred.shape[3]     # 应为 320,320
    obj_conf = pred[0, 0, :, :].sigmoid()      # (320,320)
    tx = pred[0, 1, :, :]
    ty = pred[0, 2, :, :]
    tw = pred[0, 3, :, :]
    th = pred[0, 4, :, :]

    y_grid, x_grid = torch.meshgrid(torch.arange(Hf), torch.arange(Wf), indexing='ij')
    x_grid = x_grid.float().to(device)
    y_grid = y_grid.float().to(device)

    cx = (x_grid + tx) / Wf
    cy = (y_grid + ty) / Hf
    w = torch.exp(tw) / Wf
    h = torch.exp(th) / Hf

    # 转为像素坐标
    x1 = (cx - w/2) * input_size
    y1 = (cy - h/2) * input_size
    x2 = (cx + w/2) * input_size
    y2 = (cy + h/2) * input_size

    mask = obj_conf > conf_thresh
    pred_boxes = []
    if mask.any():
        x1 = x1[mask].cpu().numpy()
        y1 = y1[mask].cpu().numpy()
        x2 = x2[mask].cpu().numpy()
        y2 = y2[mask].cpu().numpy()
        confs = obj_conf[mask].cpu().numpy()
        for i in range(len(x1)):
            pred_boxes.append([x1[i], y1[i], x2[i], y2[i]])

    # 真实框
    gt_boxes = []
    for target in targets:
        cx, cy, w, h = target
        x1_gt = (cx - w/2) * input_size
        y1_gt = (cy - h/2) * input_size
        x2_gt = (cx + w/2) * input_size
        y2_gt = (cy + h/2) * input_size
        gt_boxes.append([x1_gt, y1_gt, x2_gt, y2_gt])

    # 打印调试信息
    print(f"Max confidence: {obj_conf.max().item():.4f}")
    print(f"Num pred boxes > {conf_thresh}: {len(pred_boxes)}")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 子图1：原图 + 绿色真实框（不画预测框）
    ax1 = axes[0]
    ax1.imshow(img_np)
    for box in gt_boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                 linewidth=2, edgecolor='lime', facecolor='none')
        ax1.add_patch(rect)
    ax1.set_title('Ground Truth')
    ax1.axis('off')

    # 子图2: 损失曲线
    ax2 = axes[1]
    ax2.plot(losses)
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')

    # 子图3：特征图 + 红色预测框（坐标缩小一半）
    feat_np = feat.squeeze(0).mean(dim=0).cpu().numpy()
    feat_min, feat_max = feat_np.min(), feat_np.max()
    feat_np = (feat_np - feat_min) / (feat_max - feat_min + 1e-6)
    ax3 = axes[2]
    ax3.imshow(feat_np, cmap='gray')
    for box in pred_boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1 / 2, y1 / 2), (x2 - x1) / 2, (y2 - y1) / 2,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax3.add_patch(rect)
    ax3.set_title('Feature Map with Predictions')
    ax3.axis('off')

    plt.tight_layout()
    plt.savefig('training_result.png')
    plt.show()

    # 单独保存特征图
    feat_img = (feat_np * 255).astype(np.uint8)
    Image.fromarray(feat_img).save('feature_map.png')


if __name__ == "__main__":
    main()