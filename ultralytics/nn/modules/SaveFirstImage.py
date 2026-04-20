
import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path

__all__ = ["SaveFirstImage"]
class SaveFirstImage(nn.Module):
    def __init__(self, save_dir='./saved_images', prefix='batch', batches_per_epoch=33):
        super().__init__()
        self.save_dir = Path(save_dir).resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.batches_per_epoch = batches_per_epoch
        self.batch_counter = 0
        self.epoch_counter = 0

    def forward(self, x):
        if not self.training:
            return x

        self.batch_counter += 1
        if self.batch_counter == self.batches_per_epoch:
            # 多GPU时仅主进程保存
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                if torch.distributed.get_rank() != 0:
                    return x

            self.save_dir.mkdir(parents=True, exist_ok=True)
            num_save = min(3, x.size(0))
            for i in range(num_save):
                feat = x[i].detach().cpu().numpy()  # (C, H, W)
                # 取通道均值，得到单通道热力图
                mean_map = np.mean(feat, axis=0)   # (H, W)
                # 归一化到 0-255
                mean_map = (mean_map - mean_map.min()) / (mean_map.max() - mean_map.min() + 1e-8)
                mean_map = (mean_map * 255).astype(np.uint8)
                # 应用 JET 伪彩色，转为 3 通道 BGR
                heatmap = cv2.applyColorMap(mean_map, cv2.COLORMAP_JET)
                filename = f"{self.prefix}_epoch{self.epoch_counter+1}_batch{self.batch_counter}_img{i+1}.png"
                save_path = self.save_dir / filename
                success = cv2.imwrite(str(save_path), heatmap)
                if success:
                    print(f"Saved {save_path}")
                else:
                    print(f"Failed to save {save_path}")
            self.batch_counter = 0
            self.epoch_counter += 1
        return x