import torch
import torch.nn as nn
import torch.nn.functional as F


class EFC(nn.Module):
    """官方 EFC 模块（YOLO 适配版）"""
    def __init__(self, c1, c2, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.eps = 1e-10

        # Channel Mapper
        self.conv1 = nn.Conv2d(c1, out_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(c2, out_channels, kernel_size=1, stride=1)

        # 全局空间权重
        self.conv4_global = nn.Conv2d(out_channels, 1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(out_channels)

        # 组内交互卷积（固定4组）
        self.group_interact = nn.ModuleList([
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=1, stride=1)
            for _ in range(4)
        ])

        # 可学习的缩放/偏移
        self.gamma = nn.Parameter(torch.randn(out_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(out_channels, 1, 1))

        # 门控生成器
        self.gate_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, 1, 1),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1),
        )

        # 弱特征处理：深度可分离卷积
        self.dwconv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                                padding=1, groups=out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)

        # 强特征处理：1×1 卷积
        self.conv4_strong = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)

        # 全局平均池化
        self.apt = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x1, x2 = x  # x1: 高分辨率, x2: 低分辨率（已上采样/下采样对齐）

        # ---- 1. Channel Mapper + 门控权重 ----
        global_conv1 = self.conv1(x1)
        global_conv2 = self.conv2(x2)

        bn_x1 = self.bn(global_conv1)
        bn_x2 = self.bn(global_conv2)

        weight_1 = self.sigmoid(bn_x1)
        weight_2 = self.sigmoid(bn_x2)

        X_GLOBAL = global_conv1 + global_conv2

        # ---- 2. 全局空间注意力 ----
        x_conv4 = self.conv4_global(X_GLOBAL)
        X_4_sigmoid = self.sigmoid(x_conv4)
        X_ = X_4_sigmoid * X_GLOBAL

        # ---- 3. 分组交互（4组） ----
        X_split = X_.chunk(4, dim=1)
        out_groups = []
        for i in range(4):
            g_i = self.group_interact[i](X_split[i])
            N, C_g, H, W = g_i.shape
            x_map = g_i.reshape(N, 1, -1)
            x_av = x_map / (x_map.mean(dim=2, keepdim=True) + self.eps)
            x_softmax = F.softmax(x_av, dim=-1)
            x_weighted = x_softmax.reshape(N, C_g, H, W)
            out_groups.append(X_split[i] * x_weighted)
        out = torch.cat(out_groups, dim=1)

        # ---- 4. 特征归一化（类似 GroupNorm） ----
        N, C, H, W = out.shape
        group_num = 4
        out_reshape = out.reshape(N, group_num, C // group_num, H, W)
        out_reshape = out_reshape.reshape(N, group_num, -1)
        mean = out_reshape.mean(dim=2, keepdim=True)
        std = out_reshape.std(dim=2, keepdim=True)
        out_reshape = (out_reshape - mean) / (std + self.eps)
        out_reshape = out_reshape.reshape(N, C, H, W)
        out_norm = out_reshape * self.gamma + self.beta

        # ---- 5. 强/弱特征分离 ----
        weight_x3 = self.apt(X_GLOBAL)
        reweights = self.sigmoid(weight_x3)

        x_up_1 = (reweights >= weight_1).float()
        x_up_2 = (reweights >= weight_2).float()
        x_up = x_up_1 * X_GLOBAL + x_up_2 * X_GLOBAL

        x_low_1 = (reweights < weight_1).float()
        x_low_2 = (reweights < weight_2).float()
        x_low = x_low_1 * X_GLOBAL + x_low_2 * X_GLOBAL

        x_low_up = self.dwconv(x_low)
        x_low_up = self.conv3(x_low_up)
        x_so = self.gate_generator(x_low)
        x_low_up = x_low_up * x_so

        x_high_up = self.conv4_strong(x_up)

        # ---- 6. 融合 ----
        xL = x_low_up + x_high_up
        xL = xL + out_norm
        return xL


class EFC_FPN(nn.Module):
    """
    EFC 特征金字塔网络 (参数与 BiFPN 完全一致)

    YAML 调用格式：
        - [[2, 4, 10], 1, EFC_FPN, [[256, 512, 1024], [256, 512, 1024]]]
    参数：
        channels      : 输入通道列表 [c2, c3, c4]
        out_channels  : 输出通道列表 [o2, o3, o4] (若不指定则与输入相同)
        num_layers    : (占位，保持与 BiFPN 接口兼容)
        use_refine    : (占位，保持与 BiFPN 接口兼容)
        expand_ch     : (占位，保持与 BiFPN 接口兼容)
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
        self.efc_mid = EFC(oc_mid, oc_low, oc_mid)      # P3 + P4_up → oc_mid
        self.efc_high = EFC(oc_high, oc_mid, oc_high)   # P2 + P3_up → oc_high

        # Bottom-Up EFC
        self.efc_bu_mid = EFC(oc_mid, oc_high, oc_mid)  # P3 + P2_down → oc_mid
        self.efc_bu_low = EFC(oc_low, oc_mid, oc_low)   # P4 + P3_down → oc_low

        # 上/下采样
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            features = x
        else:
            return x

        if len(features) < 3:
            return features

        p_high, p_mid, p_low = features

        # 1. Lateral Convs
        p_high_lat = self.lat_conv_high(p_high)
        p_mid_lat = self.lat_conv_mid(p_mid)
        p_low_lat = self.lat_conv_low(p_low)

        # ---- Top-Down 增强 ----
        p_low_up = self.up(p_low_lat)
        p_mid_td = self.efc_mid([p_mid_lat, p_low_up])

        p_mid_up = self.up(p_mid_td)
        p_high_td = self.efc_high([p_high_lat, p_mid_up])

        # ---- Bottom-Up 增强 ----
        p_high_down = self.down(p_high_td)
        p_mid_bu = self.efc_bu_mid([p_mid_td, p_high_down])

        p_mid_down = self.down(p_mid_bu)
        p_low_bu = self.efc_bu_low([p_low_lat, p_mid_down])

        return [p_high_td, p_mid_bu, p_low_bu]


# ================== 简单测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 模拟 P2, P3, P4 输入
    p2 = torch.randn(2, 128, 160, 160).to(device)
    p3 = torch.randn(2, 256, 80, 80).to(device)
    p4 = torch.randn(2, 512, 40, 40).to(device)
    features = [p2, p3, p4]

    # 模拟 YAML 传参：channels=[128,256,512], out_channels=[128,256,512]
    efcfpn = EFC_FPN(channels=[128, 256, 512], out_channels=[128, 256, 512]).to(device)
    print(efcfpn)

    outputs = efcfpn(features)
    names = ['P2_out', 'P3_out', 'P4_out']
    for i, out in enumerate(outputs):
        print(f"{names[i]} shape: {out.shape}")

    loss = sum(o.mean() for o in outputs)
    loss.backward()
    print("✅ Backward passed.")

    total_params = sum(p.numel() for p in efcfpn.parameters())
    print(f"Total parameters: {total_params:,}")