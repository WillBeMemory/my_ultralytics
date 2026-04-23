# File: ultralytics/nn/modules/SplitList.py
import torch.nn as nn

class SplitList(nn.Module):
    """从输入列表中提取指定索引的特征图（兼容 base_module 风格）"""
    def __init__(self, c1, c2, index=0):
        super().__init__()
        self.c2 = c2            # 输出通道（已缩放，由框架传入）
        self.index = index

    def forward(self, x):
        if isinstance(x, list):
            return x[self.index]
        raise TypeError(f"SplitList expected a list, got {type(x)}")