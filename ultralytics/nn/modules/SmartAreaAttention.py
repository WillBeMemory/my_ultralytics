import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path

__all__=['SmartAreaAttention']

from ultralytics.nn.modules.InfSA import InfSA


# ==================== MDCA ====================
class MDCA(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, 1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, 1, 1)
        self.conv_w = nn.Conv2d(mip, 1, 1)
        self.conv_d1 = nn.Conv2d(mip, 1, 1)
        self.conv_d2 = nn.Conv2d(mip, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_h = torch.mean(x, dim=3, keepdim=True)                     # (B,C,H,1)
        x_w = torch.mean(x, dim=2, keepdim=True).permute(0,1,3,2)    # (B,C,W,1)
        x_d1 = torch.rot90(x, k=-1, dims=[2,3])
        x_d1 = torch.mean(x_d1, dim=2, keepdim=True)
        x_d1 = torch.rot90(x_d1, k=1, dims=[2,3])
        x_d2 = torch.rot90(x, k=1, dims=[2,3])
        x_d2 = torch.mean(x_d2, dim=2, keepdim=True)
        x_d2 = torch.rot90(x_d2, k=-1, dims=[2,3])

        y = torch.cat([x_h, x_w, x_d1, x_d2], dim=2)                 # (B,C,H+W+H+W,1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        split_sizes = [H, W, H, W]
        x_h, x_w, x_d1, x_d2 = torch.split(y, split_sizes, dim=2)
        x_w = x_w.permute(0,1,3,2)        # (B,mip,1,W)
        x_d1 = x_d1.permute(0,1,3,2)      # (B,mip,1,W)
        x_d2 = x_d2.permute(0,1,3,2)      # (B,mip,1,W)

        a_h = torch.sigmoid(self.conv_h(x_h))   # (B,1,H,1)
        a_w = torch.sigmoid(self.conv_w(x_w))   # (B,1,1,W)
        a_d1 = torch.sigmoid(self.conv_d1(x_d1)) # (B,1,1,W)
        a_d2 = torch.sigmoid(self.conv_d2(x_d2)) # (B,1,1,W)

        a_d1 = a_d1.expand(-1, -1, H, -1)       # (B,1,H,W)
        a_d2 = a_d2.expand(-1, -1, H, -1)       # (B,1,H,W)

        importance = a_w * a_h * a_d1 * a_d2     # (B,1,H,W)
        return importance


# ==================== SparsePatternA2 ====================
class SparsePatternA2(nn.Module):
    def __init__(self, dim, sparse_top_k=0.3, l1_lambda=1e-4, use_mask=True):
        super().__init__()
        self.sparse_top_k = sparse_top_k
        self.l1_lambda = l1_lambda
        self.use_mask = use_mask
        self.mdca = MDCA(dim)
        if use_mask:
            self.sparsity_mask = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter('sparsity_mask', None)

    def forward(self, x):
        B, C, H, W = x.shape
        importance = self.mdca(x)                     # (B,1,H,W)
        imp_flat = importance.view(B, -1)             # (B, N)
        N = imp_flat.size(1)
        attn_scores = imp_flat.unsqueeze(-1) * imp_flat.unsqueeze(-2)  # (B, N, N)

        if self.use_mask:
            mask = torch.sigmoid(self.sparsity_mask).expand(B, N, N)
            attn_scores = attn_scores * mask

        k = max(1, int(N * self.sparse_top_k))
        topk_vals, topk_idx = torch.topk(attn_scores, k, dim=-1)
        sparse_attn = torch.zeros_like(attn_scores).scatter_(-1, topk_idx, topk_vals)
        attn_weights = F.softmax(sparse_attn, dim=-1)

        x_seq = x.flatten(2).transpose(1, 2)          # (B, N, C)
        out_seq = attn_weights @ x_seq                # (B, N, C)
        out = out_seq.transpose(1, 2).reshape(B, C, H, W)
        return out

    def get_l1_loss(self):
        if self.use_mask:
            return self.l1_lambda * torch.abs(self.sparsity_mask)
        else:
            return torch.tensor(0.0, device=self.sparsity_mask.device if self.sparsity_mask is not None else torch.device('cpu'))


# ==================== 修改后的 SmartAreaAttention ====================
# class SmartAreaAttention(nn.Module):
#     def __init__(self, c1, c2,n, reduction=16, sparse_top_k=0.3, l1_lambda=1e-4, mid_channels=None,
#                  save_feature_dir=None, batches_per_epoch=33):
#         super().__init__()
#         if mid_channels is None:
#             mid_channels = max(c1, c2)
#         self.mid_channels = mid_channels
#         self.mdca = MDCA(mid_channels, reduction)
#         self.sparse_attn = SparsePatternA2(mid_channels, sparse_top_k=sparse_top_k, l1_lambda=l1_lambda, use_mask=True)
#         self.alpha = nn.Parameter(torch.tensor(0.5))
#         self.in_proj = nn.Conv2d(c1, mid_channels, 1) if c1 != mid_channels else nn.Identity()
#         self.out_proj = nn.Conv2d(mid_channels, c2, 1) if mid_channels != c2 else nn.Identity()
#
#         self.save_feature_dir = save_feature_dir
#         if save_feature_dir:
#             Path(save_feature_dir).mkdir(parents=True, exist_ok=True)
#         self.batches_per_epoch = batches_per_epoch
#         self.batch_counter = 0
#         self.epoch_counter = 0
#         self.last_out = None
#         self.last_attn = None
#
#     def forward(self, x):
#         x = self.in_proj(x)
#         attn_guide = self.mdca(x)
#         x_guided = self.alpha * x + (1 - self.alpha) * (x * attn_guide)
#         out = self.sparse_attn(x_guided)
#         out = out + x
#         out = self.out_proj(out)
#
#         if self.save_feature_dir and self.training:
#             self.last_out = out.detach().clone()
#             self.last_attn = attn_guide.detach().clone()
#             self.batch_counter += 1
#
#             if self.batch_counter == self.batches_per_epoch:
#                 # 多GPU时仅主进程保存
#                 if torch.distributed.is_available() and torch.distributed.is_initialized():
#                     if torch.distributed.get_rank() != 0:
#                         return out
#                 self._save_visualization(self.last_out, self.last_attn, self.epoch_counter + 1, self.batch_counter)
#                 self.batch_counter = 0
#                 self.epoch_counter += 1
#
#         return out

    # def _save_visualization(self, feature_map, attn_guide, epoch, batch):
    #     save_path = Path(self.save_feature_dir)
    #     save_path.mkdir(parents=True, exist_ok=True)
    #
    #     num_samples = min(3, feature_map.size(0))
    #     for i in range(num_samples):
    #         # 特征图
    #         fm = feature_map[i].detach().cpu().numpy()  # (C, H, W)
    #         fm_mean = np.mean(fm, axis=0)               # (H, W)
    #         fm_img = self._normalize_to_uint8(fm_mean)
    #         fm_color = cv2.applyColorMap(fm_img, cv2.COLORMAP_JET)
    #         cv2.imwrite(str(save_path / f"feature_epoch{epoch}_batch{batch}_img{i+1}.png"), fm_color)
    #
    #         # 注意力引导图
    #         attn = attn_guide[i].detach().cpu().numpy()  # (1, H, W)
    #         attn_map = attn[0]                          # (H, W)
    #         attn_img = self._normalize_to_uint8(attn_map)
    #         attn_color = cv2.applyColorMap(attn_img, cv2.COLORMAP_JET)
    #         cv2.imwrite(str(save_path / f"attn_epoch{epoch}_batch{batch}_img{i+1}.png"), attn_color)
    #
    #     print(f"[SmartAreaAttention] Saved feature and attn for epoch {epoch}, batch {batch}")
    #
    # @staticmethod
    # def _normalize_to_uint8(arr):
    #     arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    #     return (arr * 255).astype(np.uint8)
    #
    # def get_l1_loss(self):
    #     return self.sparse_attn.get_l1_loss()

class SmartAreaAttention(nn.Module):
    def __init__(self, c1, c2,n, reduction=16, sparse_top_k=0.3, l1_lambda=1e-4,
                 num_steps=3, gamma=0.9, mid_channels=None, shortcut=True):
        super().__init__()
        if mid_channels is None:
            mid_channels = max(c1, c2)
        self.mid_channels = mid_channels
        self.shortcut = shortcut and (c1 == c2)

        # 输入输出投影
        self.in_proj = nn.Conv2d(c1, mid_channels, 1) if c1 != mid_channels else nn.Identity()
        self.out_proj = nn.Conv2d(mid_channels, c2, 1) if mid_channels != c2 else nn.Identity()

        # 引导注意力：使用 InfSA（输出特征图可直接作为引导）
        self.guide = InfSA(mid_channels, num_steps=num_steps, gamma=gamma)

        # 稀疏剪枝模块（原有 SparsePatternA2）
        self.sparse_attn = SparsePatternA2(mid_channels, sparse_top_k=sparse_top_k, l1_lambda=l1_lambda, use_mask=True)

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x = self.in_proj(x)
        # 生成引导特征（InfSA 输出）
        attn_guide = self.guide(x)          # (B, C, H, W)
        # 融合原始特征与引导特征
        x_guided = self.alpha * x + (1 - self.alpha) * (x * attn_guide)
        # 稀疏注意力剪枝
        out = self.sparse_attn(x_guided)
        out = out + x                       # 残差
        out = self.out_proj(out)
        return out

    def get_l1_loss(self):
        return self.sparse_attn.get_l1_loss()

# ==================== 测试代码 ====================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmartAreaAttention(64, 64, num_steps=3, gamma=0.8).to(device)
    x = torch.randn(2, 64, 32, 32).to(device)
    out = model(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    loss = out.mean()
    loss.backward()
    print("Backward pass completed.")