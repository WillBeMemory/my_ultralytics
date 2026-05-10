import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

from ultralytics.nn.modules import WaveletStem


# ================== 简单检测头 ==================
class SimpleDetectionHead(nn.Module):
    """一个极简的检测头，用于模拟训练"""
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, num_classes * 5, 1)  # 5 = obj + class + tx,ty,tw,th
        )

    def forward(self, x):
        return self.conv(x)

# ================== 损失函数 (简化版 MSE 定位 + BCE 置信度) ==================
def detection_loss(pred, targets, stride=2):
    """
    pred: (1, 5, H/2, W/2)  -> [obj, tx, ty, tw, th]
    targets: list of [cx, cy, w, h] 归一化坐标
    """
    B, C, Hf, Wf = pred.shape
    obj_pred = pred[:, 0:1, :, :]          # (1,1,Hf,Wf)
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

    # 使用模块计算 BCE 损失
    bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
    loss_obj = bce_loss(obj_pred, obj_target)

    mask = obj_target > 0.5
    loss_box = 0.0
    if mask.sum() > 0:
        loss_box += nn.functional.mse_loss(tx[mask], tx_target[mask], reduction='sum')
        loss_box += nn.functional.mse_loss(ty[mask], ty_target[mask], reduction='sum')
        loss_box += nn.functional.mse_loss(tw[mask], tw_target[mask], reduction='sum')
        loss_box += nn.functional.mse_loss(th[mask], th_target[mask], reduction='sum')

    return loss_obj + loss_box

# ================== 读取标注 ==================
def load_yolo_labels(label_path, img_w, img_h):
    """读取 YOLO 格式的标注文件，返回归一化的 [cx, cy, w, h] 列表"""
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
    # ---------- 配置 ----------
    image_path = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\vis\P0001_0_800_7200_8000.jpg"
    label_path = image_path.replace('.jpg', '.txt')  # 假设同目录同名 txt
    in_channels = 3
    out_channels = 16
    num_classes = 1
    lr = 0.02
    momentum = 0.9
    weight_decay = 1e-4
    input_size = 640
    epochs = 300
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---------- 加载图像 ----------
    img = Image.open(image_path).convert('RGB')
    img = img.resize((input_size, input_size), Image.BILINEAR)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,640,640)

    # ---------- 加载标注 ----------
    targets = load_yolo_labels(label_path, img.width, img.height)

    # ---------- 构建模型 ----------
    stem = WaveletStem(in_channels=in_channels, out_channels=out_channels, use_diag_init=True).to(device)
    head = SimpleDetectionHead(out_channels, num_classes).to(device)
    model = nn.Sequential(stem, head)
    model.train()

    # ---------- 优化器 ----------
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0002)

    # ---------- 训练循环 ----------
    # ---------- 训练循环 ----------
    losses = []
    scaler = torch.cuda.amp.GradScaler()  # 可选，但建议混合精度加速
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(img_tensor)
        loss = detection_loss(pred, targets, stride=2)

        # 提前检测 NaN
        if torch.isnan(loss):
            print(f"Loss is NaN at epoch {epoch + 1}, stopping training.")
            break

        loss.backward()

        # 梯度裁剪，防止爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if (epoch + 1) % 30 == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs}, Loss: {loss.item():.4f}")

    print("训练完成！")

    # ---------- 提取特征图并显示为灰度 ----------
    import matplotlib.patches as patches

    # ... 前面的训练循环与模型定义不变 ...

    # ---------- 检测并绘制结果 ----------
    model.eval()
    with torch.no_grad():
        feat = stem(img_tensor)  # (1, 64, 320, 320)
        pred = model(img_tensor)  # (1, 5, 160, 160) 注意：由于 stride=2，头输出尺寸是输入的一半？
        # 实际上 stem 输出 (1,64,320,320)，head 中的卷积层并没有下采样，所以 head 输出尺寸应为 (1,5,320,320)
        # 检查一下：head 中只有普通卷积 (3x3 padding=1) 和 1x1，因此尺寸与 stem 输出一致，即 320x320。
        # 但我们在检测损失中假设了 stride=2 (相对于输入)，所以 head 输出的 Hf,Wf = 320,320 相对于原图 640 的 stride 是 2。
        Hf, Wf = pred.shape[2], pred.shape[3]

    # 将预测解码为边界框 (xyxy) 并筛选
    conf_thresh = 0.5
    obj_conf = pred[0, 0, :, :].sigmoid()  # (320,320)
    tx = pred[0, 1, :, :]
    ty = pred[0, 2, :, :]
    tw = pred[0, 3, :, :]
    th = pred[0, 4, :, :]

    # 生成网格坐标
    y_grid, x_grid = torch.meshgrid(torch.arange(Hf), torch.arange(Wf), indexing='ij')
    y_grid = y_grid.to(pred.device)
    x_grid = x_grid.to(pred.device)

    # 解码中心坐标和宽高
    cx = (x_grid + tx) / Wf  # 归一化到 [0,1]
    cy = (y_grid + ty) / Hf
    w = torch.exp(tw) / Wf
    h = torch.exp(th) / Hf

    # 转换为 xyxy 在图像像素坐标上
    x1 = (cx - w / 2) * input_size
    y1 = (cy - h / 2) * input_size
    x2 = (cx + w / 2) * input_size
    y2 = (cy + h / 2) * input_size

    # 筛选置信度大于阈值的预测框
    mask = obj_conf > conf_thresh
    pred_boxes = []
    pred_confs = []
    if mask.any():
        x1 = x1[mask].cpu().numpy()
        y1 = y1[mask].cpu().numpy()
        x2 = x2[mask].cpu().numpy()
        y2 = y2[mask].cpu().numpy()
        confs = obj_conf[mask].cpu().numpy()
        for i in range(len(x1)):
            pred_boxes.append([x1[i], y1[i], x2[i], y2[i]])
            pred_confs.append(confs[i])

    # 加载真实标注框并转换为像素坐标
    gt_boxes = []
    for target in targets:
        cx, cy, w, h = target
        x1_gt = (cx - w / 2) * input_size
        y1_gt = (cy - h / 2) * input_size
        x2_gt = (cx + w / 2) * input_size
        y2_gt = (cy + h / 2) * input_size
        gt_boxes.append([x1_gt, y1_gt, x2_gt, y2_gt])

    # 绘制图像上带标注框和检测框的图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 子图1：输入图像 + 真实框（绿） + 预测框（红）
    ax1 = axes[0]
    ax1.imshow(img_np)
    ax1.set_title('Detections')
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='lime', facecolor='none')
        ax1.add_patch(rect)
    for box in pred_boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)
    ax1.axis('off')

    # 子图2：损失曲线
    ax2 = axes[1]
    ax2.plot(losses)
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')

    # 子图3：特征图（灰度）
    feat_np = feat.squeeze(0).mean(dim=0).cpu().numpy()
    feat_min, feat_max = feat_np.min(), feat_np.max()
    feat_np = (feat_np - feat_min) / (feat_max - feat_min + 1e-6)
    ax3 = axes[2]
    ax3.imshow(feat_np, cmap='gray')
    ax3.set_title('Feature Map (mean)')
    ax3.axis('off')

    plt.tight_layout()
    plt.savefig('training_result_with_detections.png')
    plt.show()

    # 单独保存一张带标注的检测结果图
    fig2, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img_np)
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
    for box in pred_boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    ax.axis('off')
    plt.title('Ground Truth (green) and Predictions (red)')
    plt.savefig('detection_result.png')
    plt.show()

if __name__ == "__main__":
    main()