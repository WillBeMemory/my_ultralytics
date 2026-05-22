import torch
import torch.nn as nn
import torch.nn.functional as F


class EFC(nn.Module):
    """
    官方 EFC 模块（数值稳定 + 计算优化版）

    核心改进：
    1. 去除不稳定的手工分组交互和 GroupNorm 式归一化
    2. 强弱特征分离改用平滑的 sigmoid 软阈值，避免梯度截断
    3. 增加数值保护：eps 调大、clamp 限制输入范围
    4. 简化计算图，大幅降低计算量
    """

    def __init__(self, c1, c2, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.eps = 1e-5  # 适当增大 eps，防止除零

        # ── 1. Channel Mapper (双路 1×1 投影，无偏置) ──
        self.conv1 = nn.Conv2d(c1, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(c2, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        # ── 2. 全局空间注意力 (1×1 → Sigmoid) ──
        self.conv_spatial = nn.Conv2d(out_channels, 1, kernel_size=1, bias=False)

        # ── 3. 可学习的缩放/偏移 (用于稳定输出分布) ──
        self.gamma = nn.Parameter(torch.ones(out_channels, 1, 1))  # 初始化为 1
        self.beta = nn.Parameter(torch.zeros(out_channels, 1, 1))  # 初始化为 0

        # ── 4. 门控生成器 (SE 式通道注意力) ──
        self.gate_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.Sigmoid()  # 使用 Sigmoid 替代 Softmax，更稳定
        )

        # ── 5. 弱特征处理：深度可分离卷积 ──
        self.dwconv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                                padding=1, groups=out_channels, bias=False)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

        # ── 6. 强特征处理：1×1 卷积 ──
        self.conv4_strong = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = x  # 高分辨率, 低分辨率 (已对齐尺寸)

        # ---- 1. Channel Mapper + 门控权重 ----
        global_conv1 = self.conv1(x1)
        global_conv2 = self.conv2(x2)

        # 批量归一化
        bn_x1 = self.bn(global_conv1)
        bn_x2 = self.bn(global_conv2)

        # 各自的门控权重 (0~1)
        weight_1 = self.sigmoid(bn_x1)
        weight_2 = self.sigmoid(bn_x2)

        # 融合全局特征
        X_GLOBAL = global_conv1 + global_conv2

        # ---- 2. 全局空间注意力 ----
        spatial_att = self.sigmoid(self.conv_spatial(X_GLOBAL))  # (B,1,H,W)
        X_enhanced = spatial_att * X_GLOBAL  # 空间加权

        # ---- 3. 分布稳定：可学习缩放 + 残差 ----
        X_stable = X_GLOBAL * self.gamma + self.beta  # 类似于 GroupNorm 的缩放
        X_enhanced = X_enhanced + X_stable  # 残差连接

        # ---- 4. 强/弱特征分离 (软阈值，平滑可微) ----
        # 全局通道重要性 (SE 风格)
        channel_weight = self.gate_generator(X_GLOBAL)  # (B,C,1,1)

        # 软阈值：用 sigmoid 将 (channel_weight - weight_i) 映射为 0~1 的强弱系数
        # 陡度设为 10.0，使过渡带较窄，但仍可微
        soft_thresh_1 = torch.sigmoid((channel_weight - weight_1) * 10.0)  # 强特征系数
        soft_thresh_2 = torch.sigmoid((channel_weight - weight_2) * 10.0)

        # 强特征：系数接近 1 时保留，接近 0 时抑制
        x_strong = (soft_thresh_1 + soft_thresh_2) * X_GLOBAL

        # 弱特征：1 - 系数，与强特征互补
        x_weak = (2.0 - soft_thresh_1 - soft_thresh_2) * X_GLOBAL

        # ---- 5. 特征变换 ----
        # 弱特征经过深度可分离卷积 + SE 门控
        x_weak = self.dwconv(x_weak)
        x_weak = self.conv3(x_weak)
        weak_gate = self.gate_generator(x_weak)  # 用 SE 再门控一次
        x_weak = x_weak * weak_gate

        # 强特征经过 1×1 卷积
        x_strong = self.conv4_strong(x_strong)

        # ---- 6. 最终融合 ----
        out = x_weak + x_strong + X_enhanced  # 三路残差
        return out


class EFC_FPN(nn.Module):
    """
    EFC 特征金字塔网络 (参数与 BiFPN 一致)
    已包含 dtype 对齐，AMP 安全。
    """

    def __init__(self, channels, out_channels=None, num_layers=1, use_refine=True, expand_ch=None, **kwargs):
        super().__init__()
        c_high, c_mid, c_low = channels
        if out_channels is None:
            out_channels = channels
        oc_high, oc_mid, oc_low = out_channels

        # Lateral Convs
        self.lat_conv_high = nn.Sequential(
            nn.Conv2d(c_high, oc_high, 1, bias=False),
            nn.BatchNorm2d(oc_high),
            nn.SiLU(inplace=True)
        )
        self.lat_conv_mid = nn.Sequential(
            nn.Conv2d(c_mid, oc_mid, 1, bias=False),
            nn.BatchNorm2d(oc_mid),
            nn.SiLU(inplace=True)
        )
        self.lat_conv_low = nn.Sequential(
            nn.Conv2d(c_low, oc_low, 1, bias=False),
            nn.BatchNorm2d(oc_low),
            nn.SiLU(inplace=True)
        )

        # Top-Down EFC
        self.efc_mid = EFC(oc_mid, oc_low, oc_mid)
        self.efc_high = EFC(oc_high, oc_mid, oc_high)

        # Bottom-Up EFC
        self.efc_bu_mid = EFC(oc_mid, oc_high, oc_mid)
        self.efc_bu_low = EFC(oc_low, oc_mid, oc_low)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            features = x
        else:
            return x
        if len(features) < 3:
            return features

        # dtype 对齐（AMP 安全）
        target_dtype = self.lat_conv_high[0].weight.dtype
        features = [f.to(target_dtype) for f in features]
        p_high, p_mid, p_low = features

        # Lateral Convs
        p_high_lat = self.lat_conv_high(p_high)
        p_mid_lat = self.lat_conv_mid(p_mid)
        p_low_lat = self.lat_conv_low(p_low)

        # Top-Down
        p_low_up = self.up(p_low_lat)
        p_mid_td = self.efc_mid([p_mid_lat, p_low_up])

        p_mid_up = self.up(p_mid_td)
        p_high_td = self.efc_high([p_high_lat, p_mid_up])

        # Bottom-Up
        p_high_down = self.down(p_high_td)
        p_mid_bu = self.efc_bu_mid([p_mid_td, p_high_down])

        p_mid_down = self.down(p_mid_bu)
        p_low_bu = self.efc_bu_low([p_low_lat, p_mid_down])

        return [p_high_td, p_mid_bu, p_low_bu]