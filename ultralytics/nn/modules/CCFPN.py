# ============================================
# File: ultralytics/nn/modules/CCFPN.py
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv


class CCFPN(nn.Module):
    """
    Cross-Connected Feature Pyramid Network (CC-FPN)
    基于 CASS-Det (IEEE JSTARS 2025) 论文实现。

    核心设计:
        1. 跨连接结构: 浅层细节特征与深层语义特征直接交互，不经过中间层。
        2. 空洞卷积细化: 使用空洞卷积扩大感受野，增强多尺度特征表达。
        3. P5→P3 直接跳层连接 + P3→P5 直接跳层连接。
        4. 额外的 P6 层提供全局上下文信息。

    参数:
        channels (list): 输入特征图通道数 [c3, c4, c5]
        out_channels (list, optional): 输出特征图通道数 [o3, o4, o5]
        dilation_rate (int): 空洞卷积的膨胀率，默认 3
    """

    def __init__(self, channels, out_channels=None, dilation_rate=3):
        super().__init__()
        c3, c4, c5 = channels
        if out_channels is None:
            o3, o4, o5 = c3 // 2, c4, c5
        else:
            o3, o4, o5 = out_channels

        # ===== 1. 横向连接投影层 (1x1 Conv) =====
        self.lat_p3 = nn.Conv2d(c3, o3, 1, bias=False)  # P3 -> o3
        self.lat_p4 = nn.Conv2d(c4, o4, 1, bias=False)  # P4 -> o4
        self.lat_p5 = nn.Conv2d(c5, o5, 1, bias=False)  # P5 -> o5

        # ===== 2. 跨连接投影层 (用于跳层连接) =====
        self.cross_p5_to_p3 = nn.Conv2d(o5, o3, 1, bias=False)  # P5 -> P3 (跨连接)
        self.cross_p3_to_p5 = nn.Conv2d(o3, o5, 1, bias=False)  # P3 -> P5 (跨连接)

        # ===== 3. 自顶向下 (Top-Down) 融合层 =====
        # P5 + P4 → P4_td
        self.td_p4 = nn.Sequential(
            nn.Conv2d(o5 + o4, o4, 1, bias=False),
            nn.BatchNorm2d(o4),
            nn.SiLU(inplace=True)
        )
        # P4_td + P3 → P3_td
        self.td_p3 = nn.Sequential(
            nn.Conv2d(o4 + o3, o3, 1, bias=False),
            nn.BatchNorm2d(o3),
            nn.SiLU(inplace=True)
        )

        # ===== 4. 空洞卷积细化层 =====
        # 使用空洞卷积扩大感受野，捕获更大范围的上下文
        self.diluted_p3 = nn.Sequential(
            nn.Conv2d(o3, o3, 3, padding=dilation_rate, dilation=dilation_rate, bias=False),
            nn.BatchNorm2d(o3),
            nn.SiLU(inplace=True)
        )
        self.diluted_p4 = nn.Sequential(
            nn.Conv2d(o4, o4, 3, padding=dilation_rate, dilation=dilation_rate, bias=False),
            nn.BatchNorm2d(o4),
            nn.SiLU(inplace=True)
        )
        self.diluted_p5 = nn.Sequential(
            nn.Conv2d(o5, o5, 3, padding=dilation_rate, dilation=dilation_rate, bias=False),
            nn.BatchNorm2d(o5),
            nn.SiLU(inplace=True)
        )

        # ===== 5. 自底向上 (Bottom-Up) 融合层 =====
        # P3_out + P4_td → P4_out
        self.bu_p4 = nn.Sequential(
            nn.Conv2d(o3 + o4, o4, 1, bias=False),
            nn.BatchNorm2d(o4),
            nn.SiLU(inplace=True)
        )
        # P4_out + P5 → P5_out
        self.bu_p5 = nn.Sequential(
            nn.Conv2d(o4 + o5, o5, 1, bias=False),
            nn.BatchNorm2d(o5),
            nn.SiLU(inplace=True)
        )

        # ===== 6. 输出平滑层 =====
        self.smooth_p3 = nn.Conv2d(o3, o3, 3, padding=1, bias=False)
        self.smooth_p4 = nn.Conv2d(o4, o4, 3, padding=1, bias=False)
        self.smooth_p5 = nn.Conv2d(o5, o5, 3, padding=1, bias=False)

    def forward(self, features):
        p3, p4, p5 = features

        # ===== 横向连接投影 =====
        lat_p3 = self.lat_p3(p3)  # (B, o3, H/8, W/8)
        lat_p4 = self.lat_p4(p4)  # (B, o4, H/16, W/16)
        lat_p5 = self.lat_p5(p5)  # (B, o5, H/32, W/32)

        # ===== 自顶向下 (Top-Down) =====
        # P5 → P4
        p5_up = F.interpolate(lat_p5, size=lat_p4.shape[-2:], mode='nearest')
        p4_td = self.td_p4(torch.cat([p5_up, lat_p4], dim=1))

        # P4 → P3
        p4_td_up = F.interpolate(p4_td, size=lat_p3.shape[-2:], mode='nearest')
        p3_td = self.td_p3(torch.cat([p4_td_up, lat_p3], dim=1))

        # ===== 跨连接：P5 → P3 (直接跳层) =====
        p5_cross_to_p3 = F.interpolate(lat_p5, size=lat_p3.shape[-2:], mode='nearest')
        p5_cross_to_p3 = self.cross_p5_to_p3(p5_cross_to_p3)
        p3_td = p3_td + p5_cross_to_p3  # 融合深层语义到浅层

        # ===== 空洞卷积细化 =====
        p3_refined = self.diluted_p3(p3_td)
        p4_refined = self.diluted_p4(p4_td)
        p5_refined = self.diluted_p5(lat_p5)

        # ===== 自底向上 (Bottom-Up) =====
        # P3 → P4
        p3_down = F.avg_pool2d(p3_refined, kernel_size=2, stride=2)
        p4_out = self.bu_p4(torch.cat([p3_down, p4_refined], dim=1))

        # P4 → P5
        p4_down = F.avg_pool2d(p4_out, kernel_size=2, stride=2)
        p5_out = self.bu_p5(torch.cat([p4_down, p5_refined], dim=1))

        # ===== 跨连接：P3 → P5 (直接跳层) =====
        p3_cross_to_p5 = F.avg_pool2d(p3_refined, kernel_size=4, stride=4)
        p3_cross_to_p5 = self.cross_p3_to_p5(p3_cross_to_p5)
        p5_out = p5_out + p3_cross_to_p5  # 融合浅层细节到深层

        # ===== 输出平滑 =====
        p3_out = self.smooth_p3(p3_refined)
        p4_out = self.smooth_p4(p4_out)
        p5_out = self.smooth_p5(p5_out)

        return [p3_out, p4_out, p5_out]

# ============================================
# 测试代码（直接追加到 CCFPN.py 末尾）
# ============================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 模拟不同 scale 下的输入通道（均为缩放后实际值）
    test_configs = [
        # (模型名称, 输入通道 [P3,P4,P5], 期望输出通道 [P3,P4,P5])
        ("YOLOv11n", [128, 128, 256], [64, 128, 256]),
        ("YOLOv11s", [256, 256, 512], [128, 256, 512]),
        ("YOLOv11m", [256, 512, 512], [128, 512, 512]),
    ]

    for name, in_chs, out_chs in test_configs:
        print(f"\n{'='*50}")
        print(f"Testing {name} (input channels: {in_chs})")
        print(f"Expected output channels: {out_chs}")

        # 创建随机输入特征图（Batch=2，空间尺寸与真实 Neck 一致）
        p3 = torch.randn(2, in_chs[0], 80, 80).to(device)
        p4 = torch.randn(2, in_chs[1], 40, 40).to(device)
        p5 = torch.randn(2, in_chs[2], 20, 20).to(device)
        features = [p3, p4, p5]

        # 初始化 CCFPN
        cc_fpn = CCFPN(channels=in_chs).to(device)
        outputs = cc_fpn(features)

        # 检查输出形状
        expected_shapes = [(2, out_chs[0], 80, 80), (2, out_chs[1], 40, 40), (2, out_chs[2], 20, 20)]
        for i, out in enumerate(outputs):
            print(f"  P{i+3} output shape: {out.shape} (expected {expected_shapes[i]})")
            assert out.shape == expected_shapes[i], f"Shape mismatch for P{i+3}!"

        # 参数统计
        total_params = sum(p.numel() for p in cc_fpn.parameters())
        trainable_params = sum(p.numel() for p in cc_fpn.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # 反向传播验证
        loss = sum(out.mean() for out in outputs)
        loss.backward()
        print("  Backward pass OK")

    print("\n" + "=" * 50)
    print("All tests passed! CC-FPN is ready to use.")