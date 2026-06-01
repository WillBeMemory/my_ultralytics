import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv


class ASFFLayer(nn.Module):
    """单尺度的 ASFF 融合：将其他层统一到当前层通道数并加权求和"""
    def __init__(self, in_channels_list, level, num_levels=3):
        super().__init__()
        self.level = level
        self.num_levels = num_levels
        self.ch = in_channels_list[level]  # 当前层的目标通道数

        # 为每个输入层创建 1×1 卷积，将通道数统一到 self.ch
        self.proj = nn.ModuleList([
            Conv(in_channels_list[i], self.ch, 1, act=False) for i in range(num_levels)
        ])

        # 融合后的轻量精炼卷积
        self.refine = Conv(self.ch, self.ch, 3, act=False)

    def forward(self, x_list):
        # x_list: [P2_feat, P3_feat, P4_feat]，每个特征的通道数可能不同
        target_size = x_list[self.level].shape[2:]
        resized = []
        for i, x in enumerate(x_list):
            # 空间尺寸调整
            if i < self.level:          # 更高分辨率 → 当前分辨率：下采样（最大池化）
                x = F.max_pool2d(x, kernel_size=2 ** (self.level - i))
            elif i > self.level:        # 更低分辨率 → 当前分辨率：上采样
                x = F.interpolate(x, size=target_size, mode='nearest')
            # 通道投影到当前层通道数
            x = self.proj[i](x)
            resized.append(x)

        # 沿通道维度拼接生成空间注意力权重
        stacked = torch.stack(resized, dim=-1)          # (B, C, H, W, num_levels)
        weights = F.softmax(stacked, dim=-1)            # 每个层级的权重和为 1
        fused = torch.sum(weights * stacked, dim=-1)    # 加权求和
        return self.refine(fused)


class ASFF(nn.Module):
    """
    Adaptive Spatial Feature Fusion (ASFF)
    - 完全兼容 BiFPN 的参数接口，可直接替换 YAML 中的 BiFPN
    - 对 P2、P3、P4 三个尺度分别进行自适应空间加权融合
    """
    def __init__(self, channels, out_channels=None, num_layers=1, use_refine=True, expand_ch=None):
        super().__init__()
        if out_channels is None:
            out_channels = channels[:]
        self.channels = channels
        self.out_channels = out_channels
        self.num_levels = len(channels)

        # 输入投影层：若输入通道与输出通道不同，则用 1×1 卷积对齐
        self.proj = nn.ModuleList([
            Conv(channels[i], out_channels[i], 1, act=False) if channels[i] != out_channels[i] else nn.Identity()
            for i in range(self.num_levels)
        ])

        # 为每个尺度创建一个 ASFF 融合层，传入所有层的输出通道数列表
        self.layers = nn.ModuleList([
            ASFFLayer(out_channels, i, self.num_levels) for i in range(self.num_levels)
        ])

    def forward(self, features):
        # 输入：列表 [P2, P3, P4]
        proj_features = [proj(f) for proj, f in zip(self.proj, features)]

        outputs = []
        for i, layer in enumerate(self.layers):
            outputs.append(layer(proj_features))
        return outputs


# ================== 简单测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    bs = 2
    p2 = torch.randn(bs, 128, 160, 160).to(device)
    p3 = torch.randn(bs, 256, 80, 80).to(device)
    p4 = torch.randn(bs, 512, 40, 40).to(device)

    channels = [128, 256, 512]
    out_channels = [128, 256, 512]
    model = ASFF(channels, out_channels, num_layers=1).to(device)

    outs = model([p2, p3, p4])

    expected = [
        (bs, 128, 160, 160),
        (bs, 256, 80, 80),
        (bs, 512, 40, 40)
    ]
    for i, (out, exp) in enumerate(zip(outs, expected), start=2):
        status = "✅" if out.shape == exp else "❌"
        print(f"P{i}_out: {out.shape} expected {exp} {status}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    loss = outs[-1].sum()
    loss.backward()
    print("Backward passed successfully.")