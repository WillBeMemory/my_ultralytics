import torch
import torch.nn as nn
from ultralytics.nn.modules.block import Conv, PSABlock


class C2PSA_Dilated(nn.Module):
    """
    C2PSA + 多尺度空洞卷积分支
    在标准 C2PSA 基础上增加一条并行空洞卷积路径，
    PSA 负责通道注意力建模，空洞卷积增强多尺度感受野。

    结构：
        input(c1)
            ├─ 1x1 Conv → 2c
            │   ├─ split[0]: c (恒等路径)
            │   └─ split[1]: c (PSA路径, n层 PSABlock)
            ├─ 并行：dilation 卷积分支 (c1 → c → c1)
        concat(cv2(cat(PSA_out, identity))) + dilation_branch
        output(c1)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5, dilations=(1, 3, 5)):
        super().__init__()
        assert c1 == c2, "C2PSA requires c1 == c2"
        self.c = int(c1 * e)

        # 原始 C2PSA 部分
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(
            PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64)
            for _ in range(n)
        ))

        # 空洞卷积分支：用 1×1 降维 + 多分支 3×3 dilation + 1×1 升维
        # 参数量: c1*c_*1 + sum_k(dw 3x3) + c_*c1*1 ≈ 2*c1*c_ (可控)
        self.dil_cv1 = Conv(c1, self.c, 1, 1)

        self.dil_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.c, self.c, 3, 1, padding=d, dilation=d,
                          groups=self.c, bias=False),
                nn.BatchNorm2d(self.c),
                nn.SiLU(inplace=True),
            ) for d in dilations
        ])
        self.dil_cv2 = Conv(self.c * len(dilations), c1, 1, 1, act=False)

    def forward(self, x):
        # 原始 C2PSA 路径
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        psa_out = self.cv2(torch.cat((a, b), 1))

        # 空洞卷积分支
        y = self.dil_cv1(x)
        feat_list = []
        for br in self.dil_branches:
            feat_list.append(br(y))
        dil_out = self.dil_cv2(torch.cat(feat_list, dim=1))

        # 残差融合
        return psa_out + dil_out
