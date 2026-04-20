import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RFAConv"]
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class RFAConv(nn.Module):
    """
    感受野注意力卷积 (Receptive-Field Attention Convolution)
    根据官方实现，为每个感受野区域生成独立的注意力权重。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=None, groups=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding if padding is not None else kernel_size // 2
        self.groups = groups

        # 标准卷积权重
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # 注意力生成网络：输出通道数为 kernel_size^2
        self.attn_conv = nn.Conv2d(in_channels, kernel_size * kernel_size,
                                    kernel_size=1, stride=1, groups=groups, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 生成注意力权重图
        attn = self.attn_conv(x).sigmoid()  # (B, K*K, H, W)

        # 2. 计算输出特征图的空间尺寸
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        # 3. 将注意力图下采样到与输出尺寸相同（保证与 unfold 结果的 L 匹配）
        attn = F.interpolate(attn, size=(H_out, W_out), mode='bilinear', align_corners=False)  # (B, K*K, H_out, W_out)

        # 4. 使用 unfold 提取卷积区域
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride,
                            padding=self.padding)  # (B, C*K*K, L) 其中 L = H_out * W_out
        L = x_unfold.shape[-1]

        # 5. 将注意力图展平并广播到每个通道
        attn_flat = attn.view(B, self.kernel_size * self.kernel_size, -1)  # (B, K*K, L)
        attn_expanded = attn_flat.unsqueeze(1).expand(-1, C, -1, -1)  # (B, C, K*K, L)

        # 6. 将 unfold 结果重塑为 (B, C, K*K, L)
        x_unfold = x_unfold.view(B, C, self.kernel_size * self.kernel_size, L)  # (B, C, K*K, L)

        # 7. 应用注意力权重（逐元素相乘）
        x_weighted = x_unfold * attn_expanded  # (B, C, K*K, L)

        # 8. 将加权后的特征与卷积核进行卷积（使用矩阵乘法）
        x_weighted = x_weighted.view(B, -1, L)  # (B, C*K*K, L)
        weight_flat = self.weight.view(self.out_channels, -1)  # (O, C*K*K)
        out = torch.matmul(weight_flat, x_weighted)  # (B, O, L)

        # 9. 恢复空间形状
        out = out.view(B, self.out_channels, H_out, W_out)

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        return out

    def _compute_output_shape(self, H, W):
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        return (H_out, W_out)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RFAConv(64, 64).to(device)
    x = torch.randn(2, 64, 32, 32).to(device)
    out = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    loss = out.mean()
    loss.backward()
    print("Backward OK")