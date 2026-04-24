import torch
import torch.nn as nn
from ultralytics.nn.modules.block import Conv, Bottleneck
from ultralytics.nn.modules.SSConv import SegmentSharedConv_Shuffle


class DynamicC3k2(nn.Module):
    def __init__(self, c1, c2, num_bottlenecks=2, e=0.5, shortcut=True, g=1, num_groups=4):
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道数

        # 动态分割核心：将原 cv1 替换为分段共享卷积，输出 2*c_ 通道
        self.cv1 = SegmentSharedConv_Shuffle(c1, 2 * c_, k=1, s=1, num_groups=num_groups)

        # 可配置数量的 Bottleneck 序列
        self.m = nn.ModuleList([
            Bottleneck(c_, c_, shortcut, g, e=0.5) for _ in range(num_bottlenecks)
        ])

        # 关键修正：拼接通道 = 捷径(c_) + 主路径入口(c_) + 各Bottleneck输出(num_bottlenecks*c_)
        self.cv2 = Conv((2 + num_bottlenecks) * c_, c2, 1)

    def forward(self, x):
        x = self.cv1(x)                        # 输出 2*c_ 通道，通道已被“软分配”
        y = list(x.chunk(2, dim=1))            # y[0] 捷径, y[1] 主路径入口

        # 主路径入口和后续 Bottleneck 输出全部追加到列表
        y.extend(m(y[-1]) for m in self.m)     # 注意：y[-1] 在首次循环时为入口

        return self.cv2(torch.cat(y, dim=1))

# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DynamicC3k2(256, 256, num_bottlenecks=2, e=0.25, num_groups=4).to(device)
    x = torch.randn(2, 256, 20, 20).to(device)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")  # 期望 (2, 256, 20, 20)