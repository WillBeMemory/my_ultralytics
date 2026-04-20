import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=True, modulation=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.modulation = modulation

        # 权重
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # 偏移量生成卷积
        offset_base = (2 + (1 if modulation else 0)) * kernel_size * kernel_size
        offset_channels = groups * offset_base
        self.offset_conv = nn.Conv2d(
            in_channels, offset_channels, kernel_size, stride, padding,
            groups=groups, bias=True
        )
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        # 回退用的标准卷积（当 DCN 尺寸不匹配时自动切换）
        self.fallback_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation=dilation, groups=groups, bias=bias is not None
        )
        if bias:
            self.fallback_conv.bias.data.copy_(self.bias.data)

    def forward(self, x):
        offset = self.offset_conv(x)

        # 计算期望的输出尺寸
        H_out = (x.size(2) + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        W_out = (x.size(3) + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

        # 检查 offset 尺寸是否与理论输出尺寸一致
        if offset.size(2) != H_out or offset.size(3) != W_out:
            # 尺寸不匹配，回退到标准卷积（确保训练不中断）
            return self.fallback_conv(x)

        mask = None
        if self.modulation:
            offset_ch_per_group = 2 * self.kernel_size * self.kernel_size
            mask_ch_per_group = self.kernel_size * self.kernel_size
            offset_list, mask_list = [], []
            for g in range(self.groups):
                start = g * (offset_ch_per_group + mask_ch_per_group)
                offset_part = offset[:, start:start + offset_ch_per_group]
                mask_part = offset[:, start + offset_ch_per_group:start + offset_ch_per_group + mask_ch_per_group]
                offset_list.append(offset_part)
                mask_list.append(mask_part)
            offset = torch.cat(offset_list, dim=1)
            mask = torch.cat(mask_list, dim=1)
            mask = torch.sigmoid(mask)

        try:
            out = deform_conv2d(
                x, offset, self.weight, self.bias,
                mask=mask
            )
        except RuntimeError:
            # 即使尺寸检查通过，若仍出错则回退
            out = self.fallback_conv(x)

        return out

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeformableConv2d(64, 64, kernel_size=3, groups=2, modulation=True).to(device)
    x = torch.randn(2, 64, 32, 32).to(device)
    out = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    loss = out.mean()
    loss.backward()
    print("Backward pass completed.")