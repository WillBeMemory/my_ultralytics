import torch
import torch.nn as nn
import torch.nn.functional as F


class EFCFusion(nn.Module):
    def __init__(self, c1, c2, out_channels):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(c1, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(c2, out_channels, 1, bias=False)
        self.conv4_global = nn.Conv2d(out_channels, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.eps = 1e-5  # 适当增大 eps，防止除零

        # 分组交互卷积（4 组）
        self.interact = nn.ModuleList([
            nn.Conv2d(out_channels // 4, out_channels // 4, 1, bias=False) for _ in range(4)
        ])

        # 可学习缩放/偏移
        self.gamma = nn.Parameter(torch.randn(out_channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(out_channels, 1, 1))

        # 门控生成器
        self.gate_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1),
        )

        # 弱特征处理
        self.dwconv = nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels, bias=False)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, bias=False)

        # 强特征处理
        self.conv4 = nn.Conv2d(out_channels, out_channels, 1, bias=False)

        self.apt = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x1, x2 = x

        # AMP 安全：统一 dtype
        target_dtype = self.conv1.weight.dtype
        x1 = x1.to(target_dtype)
        x2 = x2.to(target_dtype)

        # ---- 1. 双路投影 + 门控 ----
        g1 = self.conv1(x1)
        g2 = self.conv2(x2)
        w1 = self.sigmoid(self.bn(g1))
        w2 = self.sigmoid(self.bn(g2))
        X_GLOBAL = g1 + g2

        # ---- 2. 全局空间注意力 ----
        X_att = self.sigmoid(self.conv4_global(X_GLOBAL)) * X_GLOBAL

        # ---- 3. 分组交互（在 FP32 下计算 softmax，提升稳定性） ----
        X_att_fp32 = X_att.float()         # 升精度
        X_chunks = X_att_fp32.chunk(4, dim=1)
        out_chunks = []
        for i, chunk in enumerate(X_chunks):
            o = self.interact[i](chunk.to(target_dtype))  # 卷积保持混合精度
            o = o.float()                                   # 升精度计算 softmax
            N, C, H, W = o.shape
            o_flat = o.reshape(N, 1, -1)
            o_norm = o_flat / (o_flat.mean(dim=2, keepdim=True) + self.eps)
            o_weight = F.softmax(o_norm, dim=-1).reshape(N, C, H, W)
            out_chunks.append(chunk * o_weight)
        out = torch.cat(out_chunks, dim=1).to(target_dtype)   # 降回原始精度

        # ---- 4. 特征归一化（在 FP32 下计算均值/标准差） ----
        X_GLOBAL_fp32 = X_GLOBAL.float()
        N, C, H, W = out.shape
        G = 16
        out_r = out.float().reshape(N, G, C // G, H, W).reshape(N, G, -1)
        X_r   = X_GLOBAL_fp32.reshape(N, G, C // G, H, W).reshape(N, G, -1)
        mean = X_r.mean(dim=2, keepdim=True)
        std  = torch.clamp(X_r.std(dim=2, keepdim=True), min=self.eps)  # 防止 std 过小
        out_norm = ((out_r - mean) / std).reshape(N, C, H, W).to(target_dtype)
        out_norm = out_norm * self.gamma + self.beta

        # ---- 5. 强弱特征分离 ----
        reweights = self.sigmoid(self.apt(X_GLOBAL))
        # 使用 .to(target_dtype) 代替 .float()
        x_strong = ((reweights >= w1).to(target_dtype) + (reweights >= w2).to(target_dtype)) * X_GLOBAL
        x_weak   = ((reweights <  w1).to(target_dtype) + (reweights <  w2).to(target_dtype)) * X_GLOBAL

        # ---- 6. 弱特征处理 ----
        x_weak = self.conv3(self.dwconv(x_weak))
        x_weak = x_weak * self.gate_generator(x_weak)

        # ---- 7. 强特征处理 ----
        x_strong = self.conv4(x_strong)

        return x_weak + x_strong + out_norm

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 模拟两路输入特征图（如 P4 和上采样后的 P5）
    x1 = torch.randn(2, 512, 40, 40).to(device)  # 高分辨率 (P4)
    x2 = torch.randn(2, 512, 40, 40).to(device)  # 低分辨率上采样后 (P5_up)

    # 实例化 EFCFusion 模块
    model = EFCFusion(c1=512, c2=512, out_channels=512).to(device)
    print(model)

    # 前向传播
    y = model([x1, x2])
    print(f"Input shapes: {x1.shape}, {x2.shape} -> Output shape: {y.shape}")

    # 验证输出形状
    assert y.shape == (2, 512, 40, 40), f"Shape mismatch: {y.shape}"
    print("✅ Output shape verified.")

    # 梯度测试
    loss = y.mean()
    loss.backward()
    print("✅ Backward pass succeeded.")

    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")