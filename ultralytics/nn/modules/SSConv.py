import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentSharedConv_Shuffle(nn.Module):
    """
    分段共享卷积 + 跨组交互增强版

    与原始 SegmentSharedConv 相比，新增：
      1. 通道混洗 (Channel Shuffle)：将各组通道均匀混合，打破组间隔离。
      2. 轻量级逐点卷积 (1×1)：在混洗后使用非分组卷积进行轻量级的全局通道交互，
         参数量和计算量极小，但能显著提升特征表达。

    参数:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        k (int): 卷积核尺寸，默认 3
        s (int): 步长，默认 1
        p (int, optional): 填充，默认 None 则自动计算为 k//2
        num_groups (int): 分段数量，默认 4。要求 c1 和 c2 都能被 num_groups 整除。
        shuffle_groups (int): 通道混洗的分组数，默认等于 num_groups。
    """
    def __init__(self, c1, c2, k=3, s=1, p=None, num_groups=4, shuffle_groups=None):
        super().__init__()
        p = p or k // 2
        self.num_groups = num_groups
        self.c1 = c1
        self.c2 = c2

        assert c1 % num_groups == 0, f"in_channels {c1} must be divisible by num_groups {num_groups}"
        assert c2 % num_groups == 0, f"out_channels {c2} must be divisible by num_groups {num_groups}"

        c1_per_group = c1 // num_groups
        c2_per_group = c2 // num_groups

        # 1. 分组深度卷积（空间滤波）
        self.dwconv = nn.Conv2d(c1, c1, k, s, p, groups=c1, bias=False)
        self.bn_dw = nn.BatchNorm2d(c1)
        self.act_dw = nn.SiLU(inplace=True)

        # 2. 组内共享的 1×1 卷积核（分段共享）
        self.shared_kernels = nn.Parameter(
            torch.randn(num_groups, c2_per_group, c1_per_group) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(c2))

        # 3. 通道混洗（打破组间隔离）
        self.shuffle_groups = shuffle_groups if shuffle_groups else num_groups
        self.shuffle = ChannelShuffle(self.shuffle_groups)

        # 4. 轻量级全局 1×1 卷积（跨组交互，参数量极小）
        #    使用 groups=1 的标准卷积，但输入输出通道均为 c2，核为 1×1
        self.global_pw_conv = nn.Conv2d(c2, c2, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(c2)
        self.act_pw = nn.SiLU(inplace=True)

    def forward(self, x):
        # Step 1: 分组深度卷积
        x = self.dwconv(x)
        x = self.bn_dw(x)
        x = self.act_dw(x)          # (B, c1, H, W)

        B, C, H, W = x.shape
        # Step 2: 按通道分组
        groups = torch.chunk(x, self.num_groups, dim=1)

        outputs = []
        for i, g in enumerate(groups):
            g_flat = g.reshape(B, C // self.num_groups, -1)
            # 组内共享 1×1 卷积
            out_flat = torch.matmul(self.shared_kernels[i], g_flat)
            out = out_flat.reshape(B, self.c2 // self.num_groups, H, W)
            outputs.append(out)

        # Step 3: 拼接各组
        out = torch.cat(outputs, dim=1)   # (B, c2, H, W)

        # Step 4: 偏置
        out = out + self.bias.view(1, -1, 1, 1)

        # Step 5: 通道混洗（跨组信息交互）
        out = self.shuffle(out)

        # Step 6: 轻量级全局 1×1 卷积
        out = self.global_pw_conv(out)
        out = self.bn_pw(out)
        out = self.act_pw(out)

        return out


class ChannelShuffle(nn.Module):
    """通道混洗：将输入通道分组后均匀混合"""
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, self.groups, C // self.groups, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        return x.reshape(B, C, H, W)


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SegmentSharedConv_Shuffle(128, 256, k=3, s=2, num_groups=4).to(device)
    x = torch.randn(2, 128, 160, 160).to(device)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")   # 期望 (2, 256, 80, 80)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

