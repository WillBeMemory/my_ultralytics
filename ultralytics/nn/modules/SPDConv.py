import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SPDConv(nn.Module):
    """
    可配置的 SPD 下采样模块，支持注意力引导。

    参数:
        c1           : 输入通道数
        c2           : 输出通道数 (仅在 compress=True 时生效)
        factor       : 下采样倍率，默认 2
        compress     : 是否用 1×1 卷积压缩通道。若为 False，则只做 pixel_unshuffle，输出通道 = c1 * factor²
        kernel_size  : 压缩卷积的核大小，默认 1
        pass_through : 是否在卷积后加入残差连接 (仅当 c1*factor² == c2 且 compress=True 时有效)
        use_attention: 是否启用注意力引导 (空间软加权)
        attn_bias    : 注意力模块初始偏置 (sigmoid前)，正数可提高初始权重，默认 2.0 (sig≈0.88)
    """

    def __init__(
            self, c1, c2, factor=2, compress=True, kernel_size=1,
            pass_through=False, use_attention=False, attn_bias=2.0
    ):
        super().__init__()
        self.factor = factor
        self.compress = compress
        self.use_attention = use_attention
        expanded_ch = c1 * factor ** 2

        # ----- 注意力分支 -----
        if use_attention:
            # 轻量级空间注意力：1x1 + sigmoid，参数量极小
            self.attn_conv = nn.Conv2d(c1, 1, kernel_size=1, bias=True)
            # 初始化偏置使初始权重接近 1，减少训练初期特征损失
            nn.init.constant_(self.attn_conv.bias, attn_bias)
            nn.init.xavier_uniform_(self.attn_conv.weight)
        else:
            self.attn_conv = None

        # ----- 无损空间到深度 -----
        self.space_to_depth = nn.PixelUnshuffle(downscale_factor=factor)

        # ----- 可选的 1×1 压缩 -----
        if compress:
            self.conv = nn.Sequential(
                nn.Conv2d(expanded_ch, c2, kernel_size=kernel_size,
                          padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(inplace=True)
            )
            self.pass_through = pass_through and (expanded_ch == c2)
            if self.pass_through:
                self.shortcut = nn.Identity()
        else:
            self.conv = nn.Identity()
            self.pass_through = False

    def forward(self, x):
        # 1. 注意力加权 (可选)
        if self.use_attention and self.attn_conv is not None:
            # 生成空间权重图 (B,1,H,W)，值域 [0,1]
            weight = self.attn_conv(x).sigmoid()
            x = x * weight
        else:
            weight = None  # 仅用于调试/可视化

        # 2. SPD 下采样
        out = self.space_to_depth(x)  # (B, C*F^2, H/F, W/F)

        # 3. 通道压缩
        out = self.conv(out)

        # 4. 残差连接 (若配置)
        if self.pass_through:
            out = out + self.shortcut(x) if self.shortcut else out

        return out


# ================== 辅助损失函数 (训练时可调用) ==================
def compute_attention_loss(
        weight_maps: torch.Tensor,  # (B, 1, H, W) 预测的空间权重
        target_masks: torch.Tensor,  # (B, 1, H, W) 由标注框生成的高斯/二值 mask (值域 [0,1])
        loss_type: str = 'mse',
        entropy_weight: float = 0.01  # 熵正则项，防止所有像素权重全 0 或全 1
) -> torch.Tensor:
    """
    计算注意力引导的损失，使网络关注目标区域并保留一定背景信息。

    target_masks 可通过函数 `generate_mask_from_boxes` 提前生成。
    """
    # 逐像素损失
    if loss_type == 'mse':
        pixel_loss = F.mse_loss(weight_maps, target_masks, reduction='mean')
    elif loss_type == 'bce':
        pixel_loss = F.binary_cross_entropy(weight_maps, target_masks, reduction='mean')
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    # 熵正则：防止所有权重退化为 0 或 1，保持一定的不确定性
    eps = 1e-6
    entropy = - (weight_maps * torch.log(weight_maps + eps) +
                 (1 - weight_maps) * torch.log(1 - weight_maps + eps)).mean()
    # 我们希望熵不要太小，因此用负熵惩罚（即最小化负熵 → 最大化熵）
    reg = -entropy * entropy_weight

    return pixel_loss + reg


def generate_mask_from_boxes(
        boxes,  # list of tensors, 每个元素为 (N,4) xyxy 归一化坐标 (0~1) 或像素坐标
        img_h, img_w,  # 图像原始尺寸
        feat_h, feat_w,  # 特征图尺寸 (即 weight_maps 的空间大小)
        sigma_ratio=0.05  # 高斯 sigma 相对于短边的比例
):
    """
    根据标注框生成与特征图对应的空间权重 mask (高斯加权)。

    返回: (B, 1, feat_h, feat_w) 值域 [0,1] 的 tensor
    """
    B = len(boxes)
    masks = torch.zeros(B, 1, feat_h, feat_w)
    # 缩放比例
    scale_y, scale_x = feat_h / img_h, feat_w / img_w
    sigma_y = sigma_ratio * feat_h
    sigma_x = sigma_ratio * feat_w

    for b in range(B):
        if boxes[b] is None or len(boxes[b]) == 0:
            continue
        # 框坐标转换为特征图坐标
        bx = boxes[b]  # (N,4) xyxy
        # 归一化坐标转特征图坐标
        if bx.max() <= 1.0:  # 归一化假设
            bx[:, [0, 2]] *= feat_w
            bx[:, [1, 3]] *= feat_h
        else:
            bx[:, [0, 2]] = bx[:, [0, 2]] * scale_x
            bx[:, [1, 3]] = bx[:, [1, 3]] * scale_y

        for i in range(len(bx)):
            x1, y1, x2, y2 = bx[i].tolist()
            # 中心点
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
            sigma_x_i = max(sigma_x, w * 0.3)  # 至少覆盖框宽的一部分
            sigma_y_i = max(sigma_y, h * 0.3)

            # 生成网格
            ys = torch.arange(0, feat_h, device=masks.device).float().view(-1, 1)
            xs = torch.arange(0, feat_w, device=masks.device).float().view(1, -1)
            gauss = torch.exp(-((xs - cx) ** 2) / (2 * sigma_x_i ** 2) - ((ys - cy) ** 2) / (2 * sigma_y_i ** 2))
            masks[b, 0] = torch.maximum(masks[b, 0], gauss)

    return masks


# ================== 测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 基础测试：无注意力
    x = torch.randn(2, 64, 80, 80).to(device)
    spd = SPDConv(64, 128, factor=2, compress=True, use_attention=False).to(device)
    out = spd(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")  # (2,128,40,40)

    # 注意力测试
    spd_attn = SPDConv(64, 128, factor=2, compress=True, use_attention=True).to(device)
    out_attn = spd_attn(x)
    print(f"With attention: {out_attn.shape}")

    # 辅助损失演示
    weight_maps = torch.rand(2, 1, 80, 80).to(device)  # 模拟预测权重
    target_masks = torch.rand(2, 1, 80, 80).to(device)  # 模拟目标 mask
    loss = compute_attention_loss(weight_maps, target_masks)
    print(f"Attention loss: {loss.item():.4f}")

    # 生成 target mask 示例
    boxes = [torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.7, 0.8]]),
             torch.tensor([[0.05, 0.1, 0.15, 0.2]])]
    masks = generate_mask_from_boxes(boxes, 640, 640, 80, 80, sigma_ratio=0.05)
    print(f"Generated masks shape: {masks.shape}")  # (2,1,80,80)

    print("All tests passed.")