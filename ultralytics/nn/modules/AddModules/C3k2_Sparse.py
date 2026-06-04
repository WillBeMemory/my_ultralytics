import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import Conv, C2f, C3k2, Bottleneck, C3k
try:
    from ultralytics.nn.modules.block import PSABlock
except ImportError:
    PSABlock = nn.Identity

try:
    import spconv.pytorch as spconv
except ImportError:
    import warnings
    warnings.warn("spconv not installed, C3k2_Sparse will not be available.")
    spconv = None


class SpConvBlock(nn.Module):
    """局部极差掩码驱动的 spconv 稀疏卷积块"""
    def __init__(self, channels, kernel_size=3, pool_size=3,
                 thresh_ratio=0.5, temperature=5.0):
        super().__init__()
        self.channels = channels
        self.pool_size = pool_size
        self.thresh_ratio = thresh_ratio
        self.temperature = temperature

        self.sparse_net = spconv.SparseSequential(
            spconv.SubMConv3d(
                channels, channels,
                kernel_size=(1, kernel_size, kernel_size),
                padding=(0, kernel_size // 2, kernel_size // 2)
            ),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(
                channels, channels,
                kernel_size=(1, kernel_size, kernel_size),
                padding=(0, kernel_size // 2, kernel_size // 2)
            ),
        )

    def forward(self, x):
        # ---- 关键：CPU 输入直接返回，用于 YOLO 初始化时的 stride 计算 ----
        if not x.is_cuda:
            return x

        B, C, H, W = x.shape
        device = x.device

        # ---- 1. 局部极差 + 软掩码 ----
        max_pool = F.max_pool2d(x, self.pool_size, stride=1, padding=self.pool_size // 2)
        min_pool = -F.max_pool2d(-x, self.pool_size, stride=1, padding=self.pool_size // 2)
        local_range = (max_pool - min_pool).mean(dim=1, keepdim=True)  # (B,1,H,W)

        mean_range = local_range.mean(dim=(2, 3), keepdim=True) + 1e-6
        threshold = self.thresh_ratio * mean_range
        mask_soft = torch.sigmoid(self.temperature * (local_range - threshold))

        if self.training:
            mask_hard = (mask_soft > 0.5).float()
            mask = (mask_hard - mask_soft).detach() + mask_soft  # STE
        else:
            mask = mask_soft

        # ---- 2. 提取前景像素索引与特征 ----
        mask_bool = mask.squeeze(1) > 0          # (B, H, W)
        nonzero = torch.nonzero(mask_bool)       # (N, 3)

        if nonzero.numel() == 0:
            return x

        foreground_feats = x.permute(0, 2, 3, 1)[mask_bool]  # (N, C)

        # 构造 spconv indices，确保设备正确
        batch_idx = nonzero[:, 0:1].to(device)
        yx = nonzero[:, 1:].to(device)
        zero_dim = torch.zeros(nonzero.shape[0], 1, device=device, dtype=torch.int32)
        indices = torch.cat([batch_idx, zero_dim, yx], dim=1).int()

        x_sp = spconv.SparseConvTensor(
            features=foreground_feats,
            indices=indices,
            spatial_shape=[1, H, W],
            batch_size=B
        )

        # ---- 3. 稀疏卷积 ----
        out_sp = self.sparse_net(x_sp)

        # ---- 4. 还原为稠密张量并融合 ----
        dense_out = out_sp.dense()               # (B, C, 1, H, W)
        dense_out = dense_out.squeeze(2)         # (B, C, H, W)

        out = x * (1 - mask) + dense_out * mask
        return out


class C3k2_Sparse(C3k2):
    """在 y[0] 分支加入稀疏卷积的 C3k2"""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True,
                 sparse_kernel=3, sparse_pool=3, sparse_thresh=0.5, sparse_temp=5.0):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.sparse_block = SpConvBlock(
            channels=self.c,
            kernel_size=sparse_kernel,
            pool_size=sparse_pool,
            thresh_ratio=sparse_thresh,
            temperature=sparse_temp
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y[0] = self.sparse_block(y[0])          # 透传分支应用稀疏卷积
        y.extend(m(y[-1]) for m in self.m)      # 处理分支保持不变
        return self.cv2(torch.cat(y, 1))


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = C3k2_Sparse(c1=64, c2=128, n=2, shortcut=True, sparse_thresh=0.5).to(device)
    model.train()

    x = torch.randn(2, 64, 32, 32).to(device)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    loss = y.mean()
    loss.backward()
    print("Backward passed. Test OK!")


if __name__ == "__main__":
    main()