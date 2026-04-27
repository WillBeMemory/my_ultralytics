import torch
import torch.nn as nn
import torch.nn.functional as F


class BackgroundSuppression(nn.Module):
    """
    特征图背景抑制模块（BackgroundSuppression）
    用于直接抑制特征图上的建筑物等规律纹理背景。

    参数:
        c1 (int): 输入特征图的通道数（输出通道数与输入相同）
        reduction (int): 内部融合时的通道压缩比，默认 4
    """
    def __init__(self, c1: int, reduction: int = 4):
        super().__init__()
        self.c1 = c1
        mid = max(c1 // reduction, 8)

        # ---------- 固定 Sobel 核 ----------
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]) / 8.0
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

        # ---------- 固定 Haar 小波核 ----------
        w_lh = torch.tensor([[1., -1.], [1., -1.]]) / 2.0   # 水平细节
        w_hl = torch.tensor([[1., 1.], [-1., -1.]]) / 2.0   # 垂直细节
        self.register_buffer('haar_lh', w_lh.view(1, 1, 2, 2))
        self.register_buffer('haar_hl', w_hl.view(1, 1, 2, 2))

        # ---------- 可学习的特征投影与融合 ----------
        self.feat_proj = nn.Sequential(
            nn.Conv2d(c1, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(mid + 2, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def _edge_density(self, x: torch.Tensor) -> torch.Tensor:
        """计算归一化边缘密度图 (B, 1, H, W)"""
        B, C, H, W = x.shape
        # 将 (B, C, H, W) 变形为 (B*C, 1, H, W)，然后分组卷积
        x_reshaped = x.reshape(B * C, 1, H, W)
        grad_x = F.conv2d(x_reshaped, self.sobel_x, padding=1)
        grad_y = F.conv2d(x_reshaped, self.sobel_y, padding=1)
        grad_x = grad_x.view(B, C, H, W)
        grad_y = grad_y.view(B, C, H, W)
        edge = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))   # (B, C, H, W)
        edge = edge.mean(dim=1, keepdim=True)              # (B, 1, H, W)
        # 局部对比度归一化
        edge_mean = F.avg_pool2d(edge, kernel_size=5, stride=1, padding=2)
        edge_density = edge / (edge_mean + 1e-6)
        return edge_density

    def _periodicity(self, x: torch.Tensor) -> torch.Tensor:
        """利用 Haar LH/HL 子带局部方差衡量周期性纹理 (B, 1, H, W)"""
        B, C, H, W = x.shape
        x_reshaped = x.reshape(B * C, 1, H, W)
        lh = F.conv2d(x_reshaped, self.haar_lh, stride=1, padding=0)  # (B*C, 1, H-1, W-1)
        hl = F.conv2d(x_reshaped, self.haar_hl, stride=1, padding=0)
        # 恢复尺寸
        lh = F.interpolate(lh, size=(H, W), mode='bilinear', align_corners=False)
        hl = F.interpolate(hl, size=(H, W), mode='bilinear', align_corners=False)
        lh = lh.view(B, C, H, W)
        hl = hl.view(B, C, H, W)
        # 局部方差
        lh_var = self._local_variance(lh, kernel_size=5)
        hl_var = self._local_variance(hl, kernel_size=5)
        periodicity = torch.sqrt(lh_var + hl_var + 1e-6)   # (B, C, H, W)
        periodicity = periodicity.mean(dim=1, keepdim=True) # (B, 1, H, W)
        return periodicity

    @staticmethod
    def _local_variance(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """计算局部窗口方差"""
        mean = F.avg_pool2d(x, kernel_size, stride=1, padding=kernel_size // 2)
        mean_sq = F.avg_pool2d(x.pow(2), kernel_size, stride=1, padding=kernel_size // 2)
        return (mean_sq - mean.pow(2)).clamp(min=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 纹理先验图
        edge = self._edge_density(x)       # (B, 1, H, W)
        period = self._periodicity(x)      # (B, 1, H, W)

        # 2. 原始特征投影
        feat = self.feat_proj(x)           # (B, mid, H, W)

        # 3. 融合得到空间抑制权重
        combined = torch.cat([feat, edge, period], dim=1)
        suppress_weight = self.fuse(combined)  # (B, 1, H, W)

        # 4. 背景抑制
        return x * suppress_weight


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BackgroundSuppression(c1=64).to(device)
    x = torch.randn(2, 64, 80, 80).to(device)
    out = model(x)
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    # 验证背景抑制效果
    fake = torch.zeros(1, 64, 80, 80).to(device)
    fake[..., 30:50, 30:50] = 5.0
    out_fake = model(fake)
    print("Center response > background?", out_fake[0, 0, 40, 40].item() > out_fake[0, 0, 10, 10].item())