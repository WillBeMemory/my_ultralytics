import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['DySample']
class DySample(nn.Module):
    def __init__(self, scale=2, groups=1, style='lp', offset_factor=1.0):
        super().__init__()
        self.scale = scale
        self.groups = groups
        self.style = style
        self.offset_factor = offset_factor

        if style == 'lp':
            self.offset_gen = nn.Conv2d(groups, 2 * groups, 1)
        else:
            self.offset_gen = None  # 'pl' 风格动态创建（见后文）

    def forward(self, x):
        B, C, H, W = x.shape
        dtype = x.dtype  # 关键：获取输入类型
        groups = self.groups
        scale = self.scale

        if C % groups != 0:
            raise ValueError(f"Channels {C} must be divisible by groups {groups}")

        x_group = x.view(B, groups, C // groups, H, W)

        if self.style == 'lp':
            feat = x_group.mean(dim=(2, 3, 4))  # (B, groups)
            offset = self.offset_gen(feat.unsqueeze(-1).unsqueeze(-1))  # (B, 2*groups, 1, 1)
            offset = offset.view(B, groups, 2, 1, 1).expand(-1, -1, -1, H, W)  # (B, groups, 2, H, W)
        else:
            # 'pl' 风格简化版：仅作示例，实际推荐使用官方实现
            # 注意：此处省略了复杂的卷积，建议先用 'lp' 风格
            raise NotImplementedError("'pl' style not implemented in this demo. Please use 'lp'.")

        # 生成网格时指定 dtype 和设备
        grid_x = torch.linspace(0, 1, W, device=x.device, dtype=dtype)
        grid_y = torch.linspace(0, 1, H, device=x.device, dtype=dtype)
        grid_xx, grid_yy = torch.meshgrid(grid_x, grid_y, indexing='xy')
        grid = torch.stack((grid_xx, grid_yy), dim=-1)  # (H, W, 2)
        grid = grid.unsqueeze(0).unsqueeze(0).expand(B, groups, H, W, 2)  # (B, groups, H, W, 2)

        offset_scale = self.scale * self.offset_factor
        grid = grid + offset.permute(0,1,3,4,2) * offset_scale

        # 将网格归一化到 [-1,1] 并确保类型一致
        grid = 2 * grid - 1

        # 重排输入
        x_reshaped = x_group.view(B * groups, C // groups, H, W)
        grid_reshaped = grid.view(B * groups, H, W, 2)

        # grid_sample 会保留输入的数据类型
        out = F.grid_sample(x_reshaped, grid_reshaped, mode='bilinear', align_corners=False)
        out = out.view(B, groups, C // groups, H, W).view(B, C, H, W)

        # 最后上采样
        out = F.interpolate(out, scale_factor=scale, mode='bilinear', align_corners=False)

        return out

# 测试代码
if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)
    upsample = DySample(scale=2, groups=4, style='lp')
    y = upsample(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")  # 应输出 (2,64,64,64)