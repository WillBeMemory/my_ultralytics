import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


# ---------- 轻量星操作块 ----------
class StarBlock(nn.Module):
    def __init__(self, c1, c2, k=3, e=0.5, shortcut=False, act=nn.ReLU6(inplace=True)):
        super().__init__()
        hidden = max(1, int(c1 * e))
        self.add = shortcut and c1 == c2
        self.dw1 = nn.Conv2d(c1, c1, k, padding=k // 2, groups=c1, bias=False)
        self.pw1 = nn.Conv2d(c1, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act1 = act
        self.dw2 = nn.Conv2d(c1, c1, k, padding=k // 2, groups=c1, bias=False)
        self.pw2 = nn.Conv2d(c1, hidden, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden, c2, 1, bias=False),
            nn.BatchNorm2d(c2)
        )
        self.out_act = act

    def forward(self, x):
        x1 = self.dw1(x); x1 = self.pw1(x1); x1 = self.bn1(x1); x1 = self.act1(x1)
        x2 = self.dw2(x); x2 = self.pw2(x2); x2 = self.bn2(x2)
        out = x1 * x2
        out = self.fusion(out)
        out = self.out_act(out)
        return out + x if self.add else out


# ====================== IWConv（重要性分级，按计算量排序） ======================
class IWConv(nn.Module):
    """
    基于 L2 范数的重要性加权特征精炼模块，根据重要性分级执行不同复杂度的卷积。
    分支按计算量从低到高排列：
        1. 深度可分离卷积 (DWConv)
        2. GhostConv
        3. GSConv (并行标准卷积 + 深度可分离卷积，通道混洗)
        4. 标准卷积 (Single 3x3 Conv)
        5. StarBlock (星操作)
        6. 双重标准卷积 (Two 3x3 Conv)
        7. 三重标准卷积 (Three 3x3 Conv)

    参数:
        c1, c2     : 输入/输出通道数
        n          : 占位参数（兼容框架）
        k          : 卷积核大小，默认 3
        e          : 隐藏通道扩展比，默认 0.5
        num_levels : 重要性分级数量，默认为 7，可设为较小值
    """
    def __init__(self, c1, c2, n=1, k=3, e=0.5, num_levels=5):
        super().__init__()
        self.num_levels = num_levels
        hidden = max(1, int(c1 * e))

        # ---------- 各重要性级别的卷积分支（按计算量升序） ----------
        self.branches = nn.ModuleList()

        # 1. 深度可分离卷积（DWConv）– 最轻量
        self.branches.append(nn.Sequential(
            nn.Conv2d(c1, c1, k, padding=k//2, groups=c1, bias=False),
            nn.Conv2d(c1, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True)
        ))

        # 2. GhostConv（简化版：1×1 压缩 → 3×3 DW → 拼接）
        self.branches.append(nn.Sequential(
            nn.Conv2d(c1, hidden // 2, 1, bias=False),
            nn.BatchNorm2d(hidden // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden // 2, hidden // 2, k, padding=k//2, groups=hidden // 2, bias=False),
            nn.BatchNorm2d(hidden // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden // 2, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True)
        ))

        # 3. GSConv（修正为并行双分支 + 通道混洗）
        self.branches.append(self._make_gsconv_parallel(c1, hidden, k))

        # 4. 标准 3×3 卷积（Single Conv）
        self.branches.append(nn.Sequential(
            nn.Conv2d(c1, hidden, k, padding=k//2, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True)
        ))

        # 5. StarBlock（星操作，两个DWConv → 逐元素乘法）
        self.branches.append(StarBlock(c1, hidden, k=k, e=0.5, shortcut=False))

        # 6. 双重标准卷积（Two Conv）
        self.branches.append(nn.Sequential(
            nn.Conv2d(c1, hidden, k, padding=k//2, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, k, padding=k//2, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True)
        ))

        # 7. 三重标准卷积（Three Conv）
        self.branches.append(nn.Sequential(
            nn.Conv2d(c1, hidden, k, padding=k//2, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, k, padding=k//2, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, k, padding=k//2, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True)
        ))

        self.branches = self.branches[:num_levels]

        # ---------- 分级映射层 ----------
        self.level_mapper = nn.Sequential(
            nn.Conv2d(1, num_levels, 1, bias=False),
            nn.Softmax(dim=1)
        )

        # ---------- 最终融合输出 ----------
        self.out_conv = Conv(hidden, c2, k=1, act=True)

    @staticmethod
    def _make_gsconv_parallel(c1, c2, k):
        """并行双分支 GSConv：一个标准卷积 + 一个 DWConv-1x1，拼接后通道混洗"""
        class ParallelGSConv(nn.Module):
            def __init__(self, c1, c2, k):
                super().__init__()
                self.branch1 = Conv(c1, c2 // 2, k, 1, act=True)
                self.branch2 = nn.Sequential(
                    nn.Conv2d(c1, c1, k, 1, padding=k//2, groups=c1, bias=False),
                    nn.BatchNorm2d(c1),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(c1, c2 // 2, 1, bias=False),
                    nn.BatchNorm2d(c2 // 2),
                    nn.SiLU(inplace=True)
                )
                # 通道混洗：用一个1x1卷积融合两个分支
                self.shuffle = nn.Conv2d(c2, c2, 1, bias=False)

            def forward(self, x):
                x1 = self.branch1(x)
                x2 = self.branch2(x)
                out = torch.cat([x1, x2], dim=1)
                return self.shuffle(out)
        return ParallelGSConv(c1, c2, k)

    def _compute_importance(self, x):
        importance = torch.norm(x, p=2, dim=1, keepdim=True)
        mean = F.avg_pool2d(importance, kernel_size=5, stride=1, padding=2)
        importance = importance / (mean + 1e-6)
        return importance.sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        importance = self._compute_importance(x)
        level_weights = self.level_mapper(importance)      # (B, num_levels, H, W)
        outputs = [branch(x) for branch in self.branches]  # 每个分支输出 (B, hidden, H, W)

        out = torch.zeros_like(outputs[0])
        for i, branch_out in enumerate(outputs):
            out = out + branch_out * level_weights[:, i:i+1]

        return self.out_conv(out)


# ====================== 测试 ======================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for lv in [3, 5, 7]:
        model = IWConv(128, 256, n=1, k=3, e=0.5, num_levels=lv).to(device)
        x = torch.randn(2, 128, 40, 40).to(device)
        y = model(x)
        print(f"num_levels={lv}, 输入: {x.shape} -> 输出: {y.shape}, "
              f"参数量: {sum(p.numel() for p in model.parameters()):,}")