import torch
import torch.nn as nn
import torch.nn.functional as F


class BackgroundSuppression(nn.Module):
    def __init__(self, c1: int, reduction: int = 4):
        super().__init__()
        self.c1 = c1
        mid = max(c1 // reduction, 8)

        # ---------- 固定 Sobel 核 ----------
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]) / 8.0
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(c1, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(c1, 1, 1, 1))

        # ---------- 固定 Haar 小波核 ----------
        w_lh = torch.tensor([[1., -1.], [1., -1.]]) / 2.0
        w_hl = torch.tensor([[1., 1.], [-1., -1.]]) / 2.0
        self.register_buffer('haar_lh', w_lh.view(1, 1, 2, 2).repeat(c1, 1, 1, 1))
        self.register_buffer('haar_hl', w_hl.view(1, 1, 2, 2).repeat(c1, 1, 1, 1))

        # ---------- 可学习的特征投影与融合 ----------
        self.feat_proj = nn.Sequential(
            nn.Conv2d(c1, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True)
        )

        # 融合部分，将 mid+2 映射到 1 通道
        self.fuse_conv1 = nn.Conv2d(mid + 2, mid, 3, padding=1, bias=False)
        self.fuse_bn = nn.BatchNorm2d(mid)
        self.fuse_act = nn.SiLU(inplace=True)
        self.fuse_conv2 = nn.Conv2d(mid, 1, 1, bias=True)  # 注意这里 bias=True

        # 初始化策略：让 Sigmoid 的输出初始接近 1（抑制权重为 1，即不抑制）
        nn.init.constant_(self.fuse_conv2.bias, 2.0)  # Sigmoid(2.0) ≈ 0.88，接近 1
        # 也可以使用 3.0，Sigmoid(3.0)≈0.95

        self.sigmoid = nn.Sigmoid()

    def _edge_density(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        grad_x = F.conv2d(x, self.sobel_x, padding=1, groups=C)
        grad_y = F.conv2d(x, self.sobel_y, padding=1, groups=C)
        edge = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))
        edge = edge.mean(dim=1, keepdim=True)
        edge_mean = F.avg_pool2d(edge, kernel_size=5, stride=1, padding=2)
        return edge / (edge_mean + 1e-6)

    def _periodicity(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        lh = F.conv2d(x, self.haar_lh, groups=C)  # (B, C, H-1, W-1)
        hl = F.conv2d(x, self.haar_hl, groups=C)
        lh = F.interpolate(lh, size=(H, W), mode='bilinear', align_corners=False)
        hl = F.interpolate(hl, size=(H, W), mode='bilinear', align_corners=False)
        lh_var = F.avg_pool2d(lh.pow(2), 5, 1, padding=2) - F.avg_pool2d(lh, 5, 1, padding=2).pow(2)
        hl_var = F.avg_pool2d(hl.pow(2), 5, 1, padding=2) - F.avg_pool2d(hl, 5, 1, padding=2).pow(2)
        periodicity = torch.sqrt(lh_var.clamp(min=0) + hl_var.clamp(min=0) + 1e-6)
        return periodicity.mean(dim=1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge = self._edge_density(x)
        period = self._periodicity(x)
        feat = self.feat_proj(x)

        combined = torch.cat([feat, edge, period], dim=1)

        out = self.fuse_conv1(combined)
        out = self.fuse_bn(out)
        out = self.fuse_act(out)
        out = self.fuse_conv2(out)      # 未经过 Sigmoid 的 logits
        suppress_weight = self.sigmoid(out)  # [0, 1]

        # 在训练早期，由于 bias 初始化为正，suppress_weight 接近 1，特征几乎不变
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