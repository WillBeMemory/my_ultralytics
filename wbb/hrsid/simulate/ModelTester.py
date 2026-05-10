import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from torch import ops

from ultralytics.nn.modules import HWD


# ================== 简单检测头 ==================
class SimpleDetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 5, 1)
        )
        # obj 通道偏置初始化
        nn.init.constant_(self.conv[-1].bias[0], -4.6)  # sigmoid(-4.6) ≈ 0.01
        # 其余 tx,ty,tw,th 保持默认 0，符合预期

    def forward(self, x):
        return self.conv(x)


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


# ================== 模块测试工具类 ==================
class ModuleTester:
    def __init__(self, stem, stem_out_channels, image_path, label_path,
                 input_size=640, device='cuda'):
        """
        stem              : 要测试的模块（如 HWD，输出尺寸为 input_size / downscale）
        stem_out_channels : stem 的输出通道数
        image_path        : 图像路径
        label_path        : YOLO 标注文件路径
        input_size        : 输入尺寸（正方形）
        device            : 训练设备
        """
        self.device = device
        self.input_size = input_size

        # 加载图像与标签
        self.img_tensor, self.img_np, self.targets = self._load_data(image_path, label_path)

        # 构建完整模型：stem + 检测头
        self.stem = stem.to(device)
        self.head = SimpleDetectionHead(stem_out_channels).to(device)
        self.model = nn.Sequential(self.stem, self.head)
        self.model.train()

        # 通过一次前向计算输出特征图尺寸和 stride
        with torch.no_grad():
            dummy = self.stem(self.img_tensor)
        self.feat_h, self.feat_w = dummy.shape[2], dummy.shape[3]
        self.stride = input_size / self.feat_h   # 下采样倍数

    def _load_data(self, image_path, label_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        targets = load_yolo_labels(label_path, img.width, img.height)
        return img_tensor, img_np, targets

    # ---------- 损失函数 ----------
    @staticmethod
    def detection_loss(pred, targets, feat_h, feat_w, input_size):
        """pred: (1, 5, Hf, Wf)"""
        B, C, Hf, Wf = pred.shape
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

        if len(targets) == 0:
            return F.binary_cross_entropy_with_logits(obj_pred, obj_target)

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

        loss_obj = F.binary_cross_entropy_with_logits(obj_pred, obj_target)
        mask = obj_target > 0.5
        loss_box = 0.0
        if mask.sum() > 0:
            loss_box += F.mse_loss(tx[mask], tx_target[mask])
            loss_box += F.mse_loss(ty[mask], ty_target[mask])
            loss_box += F.mse_loss(tw[mask], tw_target[mask])
            loss_box += F.mse_loss(th[mask], th_target[mask])
        return loss_obj + loss_box

    # ---------- 训练 ----------
    def train(self, epochs=300, lr=0.005, momentum=0.9, weight_decay=1e-4, clip_grad=10.0):
        optimizer = optim.SGD(self.model.parameters(), lr=lr,
                              momentum=momentum, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.001)
        self.losses = []

        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = self.model(self.img_tensor)
            loss = self.detection_loss(pred, self.targets,
                                       self.feat_h, self.feat_w, self.input_size)
            if torch.isnan(loss):
                print(f"Loss is NaN at epoch {epoch+1}. Stopping.")
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad)
            optimizer.step()
            scheduler.step()
            self.losses.append(loss.item())
            if (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}, Loss: {loss.item():.4f}")
        print("训练完成！")

    # ---------- 解码预测框 ----------
    def _decode_predictions(self, conf_thresh=0.1, iou_thresh=0.5):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.img_tensor)
        Hf, Wf = pred.shape[2], pred.shape[3]
        obj_conf = pred[0, 0, :, :].sigmoid()  # (Hf, Wf)
        tx = pred[0, 1, :, :]
        ty = pred[0, 2, :, :]
        tw = pred[0, 3, :, :]
        th = pred[0, 4, :, :]

        y_grid, x_grid = torch.meshgrid(torch.arange(Hf), torch.arange(Wf), indexing='ij')
        x_grid = x_grid.float().to(pred.device)
        y_grid = y_grid.float().to(pred.device)

        cx = (x_grid + tx) / Wf
        cy = (y_grid + ty) / Hf
        w = torch.exp(tw) / Wf
        h = torch.exp(th) / Hf

        x1 = (cx - w / 2) * self.input_size
        y1 = (cy - h / 2) * self.input_size
        x2 = (cx + w / 2) * self.input_size
        y2 = (cy + h / 2) * self.input_size

        # 初筛
        mask = obj_conf > conf_thresh
        if not mask.any():
            return []

        # 提取候选框和分数
        boxes = torch.stack([x1[mask], y1[mask], x2[mask], y2[mask]], dim=1)  # (N, 4)
        scores = obj_conf[mask]  # (N,)

        # 非极大值抑制
        keep = ops.nms(boxes, scores, iou_threshold=iou_thresh)
        boxes_nms = boxes[keep].cpu().numpy()

        # 转换回列表
        pred_boxes = [[boxes_nms[i, 0], boxes_nms[i, 1],
                       boxes_nms[i, 2], boxes_nms[i, 3]] for i in range(len(boxes_nms))]
        return pred_boxes

    # ---------- 可视化 ----------
    def visualize(self, conf_thresh=0.1, save_path='training_result.png'):
        pred_boxes = self._decode_predictions(conf_thresh)

        # 真实框
        gt_boxes = []
        for target in self.targets:
            cx, cy, w, h = target
            x1 = (cx - w/2) * self.input_size
            y1 = (cy - h/2) * self.input_size
            x2 = (cx + w/2) * self.input_size
            y2 = (cy + h/2) * self.input_size
            gt_boxes.append([x1, y1, x2, y2])

        # 提取 stem 输出的特征图
        self.model.eval()
        with torch.no_grad():
            feat = self.stem(self.img_tensor)
        feat_np = feat.squeeze(0).mean(dim=0).cpu().numpy()
        feat_np = (feat_np - feat_np.min()) / (feat_np.max() - feat_np.min() + 1e-6)

        # 绘制三张图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 原图 + 真实框（绿色）
        ax1 = axes[0]
        ax1.imshow(self.img_np)
        for box in gt_boxes:
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                     linewidth=2, edgecolor='lime', facecolor='none')
            ax1.add_patch(rect)
        ax1.set_title('Ground Truth')
        ax1.axis('off')

        # 损失曲线
        ax2 = axes[1]
        ax2.plot(self.losses)
        ax2.set_title('Training Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')

        # 特征图 + 预测框（红色）
        ax3 = axes[2]
        ax3.imshow(feat_np, cmap='gray')
        # 预测框需要按 stride 缩放
        scale = self.feat_h / self.input_size   # 特征图尺寸 / 原图尺寸
        for box in pred_boxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1*scale, y1*scale), (x2-x1)*scale, (y2-y1)*scale,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax3.add_patch(rect)
        ax3.set_title('Feature Map with Predictions')
        ax3.axis('off')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

        # 单独保存特征图
        feat_img = (feat_np * 255).astype(np.uint8)
        Image.fromarray(feat_img).save('feature_map.png')


# ================== 测试 HWD 模块 ==================
if __name__ == "__main__":
    # 导入 HWD（确保 hwd.py 在同级目录）


    image_path = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\vis\P0001_0_800_7200_8000.jpg"
    label_path = image_path.replace('.jpg', '.txt')

    # 构建待测试的 stem：这里使用 HWD，输入图像是 RGB 3 通道，输出 64 通道
    stem = HWD(c1=3, c2=64)

    tester = ModuleTester(stem=stem,
                          stem_out_channels=64,
                          image_path=image_path,
                          label_path=label_path,
                          input_size=640,
                          device='cuda' if torch.cuda.is_available() else 'cpu')

    tester.train(epochs=300, lr=0.005)
    tester.visualize(conf_thresh=0.1)