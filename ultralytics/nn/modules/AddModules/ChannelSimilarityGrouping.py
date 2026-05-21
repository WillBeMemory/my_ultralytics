import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelSimilarityGrouping(nn.Module):
    """
    通道相似度分组增强模块（计算量优化版）

    操作：
    1. 全局平均池化，将每个通道压缩为标量
    2. 计算通道间余弦相似度矩阵 (C, C)
    3. 根据相似度阈值将通道划分为若干组（贪心连通分量）
    4. 对每组通道进行组内聚合（均值）并强化回原通道（残差增强）
    5. 经过 Bottleneck 精炼输出

    参数：
        c1: 输入通道数
        c2: 输出通道数
        sim_thresh: 相似度阈值，高于此值视为同组 (默认 0.7)
        agg_mode: 组内聚合方式，'mean' 或 'max' (默认 'mean')
        use_residual: 是否在组内强化后添加残差连接 (默认 True)
        bottleneck_e: Bottleneck 内部扩展比 (默认 0.5)
        shortcut: Bottleneck 残差连接 (默认 True)
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        sim_thresh: float = 0.7,
        agg_mode: str = "mean",
        use_residual: bool = True,
        bottleneck_e: float = 0.5,
        shortcut: bool = True,
    ):
        super().__init__()
        self.sim_thresh = sim_thresh
        self.agg_mode = agg_mode
        self.use_residual = use_residual

        # 瓶颈精炼（轻量深度可分离卷积）
        c_ = int(c2 * bottleneck_e)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c1, c_, 1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_, c_, 3, padding=1, groups=c_, bias=False),  # 深度卷积
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def _compute_similarity(self, x):
        """
        计算通道间余弦相似度矩阵（全局池化版本）
        x: (B, C, H, W) → 返回 (C, C)
        """
        B, C, H, W = x.shape
        # 全局平均池化，每个通道压缩为一个标量
        x_pool = x.mean(dim=[2, 3])  # (B, C)
        # L2 归一化
        x_norm = F.normalize(x_pool, dim=1)  # (B, C)
        # 相似度矩阵：B × C × C，然后对 batch 取平均
        sim = torch.bmm(x_norm.unsqueeze(2), x_norm.unsqueeze(1))  # (B, C, C)
        return sim.mean(dim=0)  # (C, C)

    def _cluster_groups(self, sim_matrix):
        """
        贪心法根据阈值分组：将相似度大于 thresh 的通道合并为一组
        """
        C = sim_matrix.shape[0]
        visited = torch.zeros(C, dtype=torch.bool, device=sim_matrix.device)
        groups = []

        for i in range(C):
            if visited[i]:
                continue
            group = [i]
            visited[i] = True
            for j in range(i + 1, C):
                if not visited[j] and sim_matrix[i, j] >= self.sim_thresh:
                    group.append(j)
                    visited[j] = True
            groups.append(group)
        return groups

    def _aggregate_groups(self, x, groups):
        """
        对每组通道进行组内聚合（均值或最大值），然后以残差方式强化原始通道
        """
        B, C, H, W = x.shape
        enhanced = x.clone()

        if self.agg_mode == "mean":
            agg_fn = lambda g: enhanced[:, g, :, :].mean(dim=1, keepdim=True)
        elif self.agg_mode == "max":
            agg_fn = lambda g: enhanced[:, g, :, :].max(dim=1, keepdim=True)[0]
        else:
            raise ValueError(f"Unknown agg_mode: {self.agg_mode}")

        for group in groups:
            if len(group) <= 1:
                continue  # 单个通道不需要强化
            g = torch.tensor(group, device=x.device)
            group_feat = agg_fn(g)  # (B, 1, H, W)
            # 将聚合特征加回组内每个通道（缩放因子 0.1 防止过强）
            for idx in group:
                enhanced[:, idx, :, :] += group_feat.squeeze(1) * 0.1

        if self.use_residual:
            enhanced = enhanced + x
        return enhanced

    def forward(self, x):
        # 1. 计算相似度矩阵（轻量级）
        sim = self._compute_similarity(x)  # (C, C)

        # 2. 分组
        groups = self._cluster_groups(sim)

        # 3. 组内聚合与强化
        x_enhanced = self._aggregate_groups(x, groups)

        # 4. 瓶颈精炼
        out = self.bottleneck(x_enhanced)
        return out


# ================== 简单测试 ==================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 模拟 P5 特征图 (B, 256, 20, 20)
    x = torch.randn(2, 256, 20, 20).to(device)

    model = ChannelSimilarityGrouping(
        c1=256, c2=256, sim_thresh=0.7, agg_mode='mean', use_residual=True
    ).to(device)

    print(model)

    y = model(x)
    print(f"Input: {x.shape} → Output: {y.shape}")

    assert y.shape == (2, 256, 20, 20), f"Shape mismatch: {y.shape}"
    print("✅ Output shape verified.")

    loss = y.mean()
    loss.backward()
    print("✅ Backward passed.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")