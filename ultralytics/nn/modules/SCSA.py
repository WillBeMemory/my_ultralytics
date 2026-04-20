import typing as t
import torch
import torch.nn as nn
from einops import rearrange
# 相对导入，避免循环
from .conv import Conv          # Conv 通常定义在 conv.py 中
from .block import Bottleneck, ABlock, C3k  # Bottleneck 定义在 bottleneck.py 中（如果存在）

# 论文题目：SCSA: Exploring the Synergistic Effects Between Spatial and Channel Attention
# 中文题目:  SCSA: 探索空间注意力和通道注意力之间的协同效应
# 论文链接：https://arxiv.org/pdf/2407.05128

# 代码来源：https://github.com/HZAI-ZJNU/SCSA
# 代码整理与注释：公众号：AI缝合术
"""
2024年全网最全即插即用模块,全部免费!包含各种卷积变种、最新注意力机制、特征融合模块、上下采样模块，
适用于人工智能(AI)、深度学习、计算机视觉(CV)领域，适用于图像分类、目标检测、实例分割、语义分割、
单目标跟踪(SOT)、多目标跟踪(MOT)、红外与可见光图像融合跟踪(RGBT)、图像去噪、去雨、去雾、去模糊、超分等任务，
模块库持续更新中......
"""


# AI缝合术github：https://github.com/AIFengheshu/Plug-play-modules

_all_ = ["A2C2f_SCSA","C2PSA_SCSA","SCSA"]
class A2C2f_SCSA(nn.Module):
    """A2C2f with SCSA attention at the end."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
        # SCSA parameters (all have defaults for compatibility)
        head_num: int = 4,
        window_size: int = 7,
        group_kernel_sizes = [3, 5, 7, 9],
        qkv_bias: bool = False,
        fuse_bn: bool = False,
        down_sample_mode: str = 'avg_pool',
        attn_drop_ratio: float = 0.,
        gate_layer: str = 'sigmoid',
    ):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        if a2:
            # ABlock requires channel dimension divisible by 32
            assert c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        # Optional residual scaling factor (used only when a2 and residual are True)
        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None

        # Build inner modules (ABlock or C3k)
        self.m = nn.ModuleList()
        for _ in range(n):
            if a2:
                # Two ABlocks in sequence as in original A2C2f
                block = nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            else:
                # Use C3k (note: g may be -1, but C3k should handle it internally)
                block = C3k(c_, c_, 2, shortcut, g)
            self.m.append(block)

        # Create SCSA module
        self.scsa = SCSA(
            dim=c2,
            head_num=head_num,
            window_size=window_size,
            group_kernel_sizes=group_kernel_sizes,
            qkv_bias=qkv_bias,
            fuse_bn=fuse_bn,
            down_sample_mode=down_sample_mode,
            attn_drop_ratio=attn_drop_ratio,
            gate_layer=gate_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            out = x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
        else:
            out = y
        # Apply SCSA attention
        out = self.scsa(out)
        return out

class SCSA(nn.Module):
    def __init__(
            self,
            dim: int,
            head_num: int = 8,
            window_size: int = 7,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            down_sample_mode: str = 'avg_pool',
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'sigmoid',
    ):
        super(SCSA, self).__init__()  # 调用 nn.Module 的构造函数
        self.dim = dim  # 特征维度
        self.head_num = head_num  # 注意力头数
        self.head_dim = dim // head_num  # 每个头的维度
        self.scaler = self.head_dim ** -0.5  # 缩放因子
        self.group_kernel_sizes = group_kernel_sizes  # 分组卷积核大小
        self.window_size = window_size  # 窗口大小
        self.qkv_bias = qkv_bias  # 是否使用偏置
        self.fuse_bn = fuse_bn  # 是否融合批归一化
        self.down_sample_mode = down_sample_mode  # 下采样模式

        assert self.dim % 4 == 0, 'The dimension of input feature should be divisible by 4.'  # 确保维度可被4整除
        self.group_chans = group_chans = self.dim // 4  # 分组通道数

        # 定义局部和全局深度卷积层
        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)

        # 注意力门控层
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)  # 水平方向的归一化
        self.norm_w = nn.GroupNorm(4, dim)  # 垂直方向的归一化

        self.conv_d = nn.Identity()  # 直接连接
        self.norm = nn.GroupNorm(1, dim)  # 通道归一化
        # 定义查询、键和值的卷积层
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)  # 注意力丢弃层
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()  # 通道注意力门控

        # 根据窗口大小和下采样模式选择下采样函数
        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化
        else:
            if down_sample_mode == 'recombination':
                self.down_func = self.space_to_chans  # 重组合下采样
                # 维度降低
                self.conv_d = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)  # 平均池化
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)  # 最大池化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入张量 x 的维度为 (B, C, H, W)
        """
        # 计算空间注意力优先级
        b, c, h_, w_ = x.size()  # 获取输入的形状
        # (B, C, H)
        x_h = x.mean(dim=3)  # 沿着宽度维度求平均
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)  # 拆分通道
        # (B, C, W)
        x_w = x.mean(dim=2)  # 沿着高度维度求平均
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)  # 拆分通道

        # 计算水平注意力
        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)  # 调整形状

        # 计算垂直注意力
        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)  # 调整形状

        # 计算最终的注意力加权
        x = x * x_h_attn * x_w_attn

        # 基于自注意力的通道注意力
        # 减少计算量
        y = self.down_func(x)  # 下采样
        y = self.conv_d(y)  # 维度转换
        _, _, h_, w_ = y.size()  # 获取形状

        # 先归一化，然后重塑 -> (B, H, W, C) -> (B, C, H * W)，并生成 q, k 和 v
        y = self.norm(y)  # 归一化
        q = self.q(y)  # 计算查询
        k = self.k(y)  # 计算键
        v = self.v(y)  # 计算值
        # (B, C, H, W) -> (B, head_num, head_dim, N)
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))

        # 计算注意力
        attn = q @ k.transpose(-2, -1) * self.scaler  # 点积注意力计算
        attn = self.attn_drop(attn.softmax(dim=-1))  # 应用注意力丢弃
        # (B, head_num, head_dim, N)
        attn = attn @ v  # 加权值
        # (B, C, H_, W_)
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))
        # (B, C, 1, 1)
        attn = attn.mean((2, 3), keepdim=True)  # 求平均
        attn = self.ca_gate(attn)  # 应用通道注意力门控
        return attn * x  # 返回加权后的输入


class SCSABlock(nn.Module):
    """
    SCSA 注意力块，用于替代 C2PSA 中的 PSABlock。
    包含 SCSA 注意力和一个 3×3 卷积，支持残差连接。
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=None):
        super().__init__()
        self.scsa = SCSA(c)               # SCSA 注意力模块
        self.conv = Conv(c, c, 3)          # 可选卷积，增强局部特征
        self.shortcut = True               # 是否使用残差连接

    def forward(self, x):
        identity = x
        x = self.scsa(x)                   # 空间-通道协同注意力
        x = self.conv(x)                   # 局部特征提取
        if self.shortcut and x.shape == identity.shape:
            return x + identity
        return x


class C2PSA_SCSA(nn.Module):
    """C2PSA with SCSA attention (replaces PSABlock with SCSABlock)."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2PSA_SCSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of SCSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2, "C2PSA_SCSA requires input and output channels to be equal."
        self.c = int(c1 * e)                           # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)          # split into a and b
        self.cv2 = Conv(2 * self.c, c1, 1)             # final fusion

        # 使用 SCSABlock 替换 PSABlock
        self.m = nn.Sequential(*(SCSABlock(self.c) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)                                   # apply SCSA blocks
        return self.cv2(torch.cat((a, b), dim=1))

if __name__ == "__main__":
    # 参数: dim特征维度; head_num注意力头数; window_size = 7 窗口大小
    scsa = SCSA(dim=32, head_num=8, window_size=7)
    # 随机生成输入张量 (B, C, H, W)
    input_tensor = torch.rand(1, 32, 256, 256)
    # 打印输入张量的形状
    print(f"输入张量的形状: {input_tensor.shape}")
    # 前向传播
    output_tensor = scsa(input_tensor)
    # 打印输出张量的形状
    print(f"输出张量的形状: {output_tensor.shape}")

