import torch
import torch.nn as nn
import torch.nn.functional as F


def _deform_conv2d_pure(input, offset, weight, bias=None, stride=1, padding=0, dilation=1, mask=None):
    """Pure PyTorch DCN v2 implementation using F.grid_sample (CUDA-safe).

    Args:
        input:   (B, C_in, H_in, W_in)
        offset:  (B, 2 * kH * kW, H_out, W_out)
        weight:  (C_out, C_in, kH, kW)
        mask:    (B, kH * kW, H_out, W_out) or None
    """
    B, C_in, H_in, W_in = input.shape
    C_out, _, kH, kW = weight.shape

    # Pad input
    if padding > 0:
        input = F.pad(input, [padding] * 4)
    H, W = input.shape[2], input.shape[3]

    # Output spatial dims
    H_out = (H_in + 2 * padding - dilation * (kH - 1) - 1) // stride + 1
    W_out = (W_in + 2 * padding - dilation * (kW - 1) - 1) // stride + 1

    # Reshape offset: (B, 2*kH*kW, H_out, W_out) -> (B, kH*kW, 2, H_out, W_out)
    offset = offset.reshape(B, kH * kW, 2, H_out, W_out)

    # Regular kernel sampling positions: [0, d, 2d, ..., (k-1)d] (NOT centered)
    ky = torch.arange(kH, device=input.device, dtype=torch.float32) * dilation
    kx = torch.arange(kW, device=input.device, dtype=torch.float32) * dilation
    ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing='ij')
    kernel_y = ky_grid.reshape(1, kH * kW, 1, 1, 1)
    kernel_x = kx_grid.reshape(1, kH * kW, 1, 1, 1)

    # Output base positions
    y_out = torch.arange(H_out, device=input.device, dtype=torch.float32) * stride
    x_out = torch.arange(W_out, device=input.device, dtype=torch.float32) * stride

    # Sampling positions
    sample_y = (y_out.view(1, 1, 1, H_out, 1) + kernel_y + offset[:, :, 1:2]).expand(-1, -1, -1, -1, W_out)
    sample_x = (x_out.view(1, 1, 1, 1, W_out) + kernel_x + offset[:, :, 0:1]).expand(-1, -1, -1, H_out, -1)

    # Normalize to [-1, 1] for grid_sample
    sample_y = 2.0 * sample_y / (H - 1) - 1.0
    sample_x = 2.0 * sample_x / (W - 1) - 1.0

    # Build grid: (B * kH*kW, H_out, W_out, 2)
    grid = torch.stack([sample_x, sample_y], dim=-1).reshape(B * kH * kW, H_out, W_out, 2)

    # Expand input: (B * kH*kW, C_in, H, W)
    input_exp = input.unsqueeze(1).expand(-1, kH * kW, -1, -1, -1).reshape(B * kH * kW, C_in, H, W)

    # Sample
    sampled = F.grid_sample(input_exp, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    sampled = sampled.reshape(B, kH * kW, C_in, H_out, W_out)
    sampled = sampled.permute(0, 3, 4, 2, 1)  # (B, H_out, W_out, C_in, kH*kW)

    # Apply modulation mask
    if mask is not None:
        mask = mask.reshape(B, kH * kW, H_out, W_out).permute(0, 2, 3, 1)
        mask = torch.sigmoid(mask)
        sampled = sampled * mask.unsqueeze(-2)

    # Apply convolution weights
    weight_flat = weight.reshape(C_out, C_in, kH * kW)
    out = torch.einsum('bhwck,ock->bhwo', sampled, weight_flat)

    out = out.permute(0, 3, 1, 2)  # (B, C_out, H_out, W_out)

    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)

    return out


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

        # 回退用的标准卷积
        self.fallback_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation=dilation, groups=groups, bias=bias is not None
        )
        if bias:
            self.fallback_conv.bias.data.copy_(self.bias.data)

    def forward(self, x):
        offset = self.offset_conv(x)

        # Clamp offsets for safety
        max_offset = 3 * self.kernel_size
        offset = torch.clamp(offset, min=-max_offset, max=max_offset)

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

        # Pure PyTorch DCN (groups=1) or fallback (groups>1)
        if self.groups == 1:
            try:
                # Apply DCN per group if groups>1, else single group
                out = _deform_conv2d_pure(
                    x, offset, self.weight, self.bias,
                    stride=self.stride, padding=self.padding,
                    dilation=self.dilation, mask=mask,
                )
            except Exception:
                out = self.fallback_conv(x)
        else:
            # groups > 1 not supported in pure DCN; use fallback
            out = self.fallback_conv(x)

        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeformableConv2d(64, 64, kernel_size=3, groups=1, modulation=True).to(device)
    x = torch.randn(2, 64, 32, 32).to(device)
    out = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    loss = out.mean()
    loss.backward()
    print("Backward pass completed.")
