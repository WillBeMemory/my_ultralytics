import torch
import torch.nn as nn
import sys

# 如果需要安装 thop: pip install thop
try:
    from thop import profile
except ImportError:
    print("请先安装 thop: pip install thop")
    sys.exit(1)

def compute_flops_comparison():
    # 环境设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, C, H, W = 1, 32, 80, 80
    K = 3
    fg_ratio = 0.3  # 前景占比，与之前测试一致（约30%）
    N_active = int(fg_ratio * H * W)  # 激活点数 ≈ 1920

    # ----- 普通卷积 FLOPs 计算 -----
    dummy_input = torch.randn(B, C, H, W).to(device)
    conv_standard = nn.Conv2d(C, C, K, stride=1, padding=K//2, bias=False).to(device)
    flops_standard, params_standard = profile(conv_standard, inputs=(dummy_input,))
    # thop 返回的 FLOPs 单位是次，通常 1 MAC = 2 FLOPs，但这里直接使用 thop 的数值

    # ----- 稀疏卷积 FLOPs 估算 -----
    # 公式：FLOPs ≈ B × C_in × C_out × K^2 × N_active × 2 （乘加各一次，thop 也按乘加各一次计数）
    # 为与 thop 保持一致，我们手动计算 MACs 并乘以 2
    macs_standard = B * C * C * K * K * H * W
    macs_sparse  = B * C * C * K * K * N_active
    flops_standard_manual = macs_standard * 2   # 乘一次加一次
    flops_sparse_manual   = macs_sparse * 2

    # 结果输出
    print("=" * 60)
    print("        标准卷积 vs 稀疏卷积 计算量对比")
    print("=" * 60)
    print(f"特征图形状          : ({B}, {C}, {H}, {W})")
    print(f"卷积核尺寸          : {K}×{K}")
    print(f"前景激活点数        : {N_active} (比例 {fg_ratio})")
    print("-" * 60)
    print(f"标准卷积 FLOPs (thop): {flops_standard:.2f}")
    print(f"标准卷积 MACs        : {macs_standard:,}")
    print(f"标准卷积 FLOPs (手工): {flops_standard_manual:,}")
    print("-" * 60)
    print(f"稀疏卷积 MACs        : {macs_sparse:,}")
    print(f"稀疏卷积 FLOPs (手工): {flops_sparse_manual:,}")
    print("-" * 60)
    print(f"理论加速比 (MACs)    : {macs_standard / macs_sparse:.2f}x")
    print(f"节省计算量比例       : {(1 - macs_sparse/macs_standard) * 100:.1f}%")
    print("=" * 60)
    print("注：稀疏卷积实际加速受硬件、索引开销等影响，")
    print("    但核心计算量随前景比例线性下降。")

if __name__ == "__main__":
    compute_flops_comparison()