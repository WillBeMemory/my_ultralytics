import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Conv

class BiFPN_Add(nn.Module):
    """快速归一化加权融合，支持自定义初始权重"""
    def __init__(self, num_inputs=2, init_weights=None):
        super().__init__()
        if init_weights is None:
            self.w = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        else:
            # 必须确保传入的 init_weights 长度与 num_inputs 一致
            self.w = nn.Parameter(torch.tensor(init_weights, dtype=torch.float32))
        self.eps = 1e-4

    def forward(self, xs):
        w = F.relu(self.w)                     # 确保权重非负
        w_norm = w / (w.sum() + self.eps)      # 快速归一化融合
        return sum(w_norm[i] * xs[i] for i in range(len(xs)))


class BiFPN(nn.Module):
    """
    BiFPN 完整实现 (EfficientDet)，支持 P5 权重增强。
    参数:
        channels (tuple): 输入特征通道数 [c3, c4, c5]
        out_channels (tuple, optional): 输出特征通道数 [o3, o4, o5]。
                                        默认将 P3 通道压缩为 c3//2，P4/P5 保持不变。
        use_depthwise (bool): 是否使用深度可分离卷积 (原论文推荐)
        p5_boost (float): P5 特征增强因子的初始值，默认 1.5
    """
    def __init__(self, channels, out_channels=None, use_depthwise=True, p5_boost=1.5):
        super().__init__()
        c3, c4, c5 = channels
        if out_channels is None:
            o3, o4, o5 = c3 // 2, c4, c5
        else:
            o3, o4, o5 = out_channels

        # ---- P5 增强因子（可学习） ----
        self.p5_boost = nn.Parameter(torch.tensor(p5_boost, dtype=torch.float32))

        # ---- 通道投影 (1x1 Conv) ----
        self.p3_proj = nn.Conv2d(c3, o3, 1, bias=False)
        self.p5_to_p4 = nn.Conv2d(c5, o4, 1, bias=False)
        self.p4_to_p3 = nn.Conv2d(o4, o3, 1, bias=False)
        self.p3_to_p4 = nn.Conv2d(o3, o4, 1, bias=False)
        self.p4_to_p5 = nn.Conv2d(o4, o5, 1, bias=False)

        # ---- 加权融合节点（P5 分支初始权重更高） ----
        self.p4_td_fuse = BiFPN_Add(2, [1.0, 1.5])   # [P4权重, P5_up权重]
        self.p3_td_fuse = BiFPN_Add(2)                # 均匀初始化
        self.p4_bu_fuse = BiFPN_Add(2)                # 均匀初始化
        self.p5_bu_fuse = BiFPN_Add(2, [1.5, 1.0])   # [P5权重, P4_down权重]

        # ---- 融合后平滑卷积 ----
        if use_depthwise:
            self.smooth_p4_td = nn.Conv2d(o4, o4, 3, padding=1, groups=o4, bias=False)
            self.smooth_p3_td = nn.Conv2d(o3, o3, 3, padding=1, groups=o3, bias=False)
            self.smooth_p4_out = nn.Conv2d(o4, o4, 3, padding=1, groups=o4, bias=False)
            self.smooth_p5_out = nn.Conv2d(o5, o5, 3, padding=1, groups=o5, bias=False)
        else:
            self.smooth_p4_td = nn.Conv2d(o4, o4, 3, padding=1, bias=False)
            self.smooth_p3_td = nn.Conv2d(o3, o3, 3, padding=1, bias=False)
            self.smooth_p4_out = nn.Conv2d(o4, o4, 3, padding=1, bias=False)
            self.smooth_p5_out = nn.Conv2d(o5, o5, 3, padding=1, bias=False)

    def forward(self, features):
        p3, p4, p5 = features

        # ---- P5 增强 ----
        p5 = p5 * self.p5_boost

        # ---- 自顶向下 ----
        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        p4_td = self.smooth_p4_td(self.p4_td_fuse([p4, self.p5_to_p4(p5_up)]))

        p4_td_up = F.interpolate(p4_td, size=p3.shape[-2:], mode='nearest')
        p3_td = self.smooth_p3_td(self.p3_td_fuse([self.p3_proj(p3), self.p4_to_p3(p4_td_up)]))

        # ---- 自底向上 ----
        p3_down = F.avg_pool2d(p3_td, kernel_size=2, stride=2)
        p4_out = self.smooth_p4_out(self.p4_bu_fuse([p4_td, self.p3_to_p4(p3_down)]))

        p4_down = F.avg_pool2d(p4_out, kernel_size=2, stride=2)
        p5_out = self.smooth_p5_out(self.p5_bu_fuse([p5, self.p4_to_p5(p4_down)]))

        return [p3_td, p4_out, p5_out]

# ============================================
# 测试代码（直接追加到 BiFPN.py 末尾）
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

        # 初始化 BiFPN
        bifpn = BiFPN(channels=in_chs, use_depthwise=True).to(device)
        outputs = bifpn(features)

        # 检查输出形状
        expected_shapes = [(2, out_chs[0], 80, 80), (2, out_chs[1], 40, 40), (2, out_chs[2], 20, 20)]
        for i, out in enumerate(outputs):
            print(f"  P{i+3} output shape: {out.shape} (expected {expected_shapes[i]})")
            assert out.shape == expected_shapes[i], f"Shape mismatch for P{i+3}!"

        # 参数统计
        total_params = sum(p.numel() for p in bifpn.parameters())
        trainable_params = sum(p.numel() for p in bifpn.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # 反向传播验证
        loss = sum(out.mean() for out in outputs)
        loss.backward()
        print("  Backward pass OK")

    print("\n" + "=" * 50)
    print("All tests passed! BiFPN is ready to use.")