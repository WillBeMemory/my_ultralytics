import torch
import torch.nn as nn
import torch.nn.functional as F


class HWD(nn.Module):
    """
    可学习小波下采样模块 (HWD)
    1. 可学习 Haar 分解 → (B, 4*C_in, H/2, W/2)
    2. 1×1 卷积压缩 → (B, C_out, H/2, W/2)

    内置约束损失（正交性 + 阻带能量），通过 regularization_loss() 获取。
    """
    def __init__(self, c1: int, c2: int, act: bool = True,
                 reg_w_orth: float = 0.1, reg_w_stop: float = 0.01, reg_nyquist: float = 0.4):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        # 约束超参数
        self.reg_w_orth = reg_w_orth
        self.reg_w_stop = reg_w_stop
        self.reg_nyquist = reg_nyquist

        # ---------- 可学习的 Haar 分解滤波器 ----------
        w_ll = torch.tensor([[1., 1.],
                             [1., 1.]]) / 2.0
        w_lh = torch.tensor([[1., -1.],
                             [1., -1.]]) / 2.0
        w_hl = torch.tensor([[1., 1.],
                             [-1., -1.]]) / 2.0
        w_hh = torch.tensor([[1., -1.],
                             [-1., 1.]]) / 2.0
        haar_init = torch.stack([w_ll, w_lh, w_hl, w_hh], dim=0).unsqueeze(1)  # (4, 1, 2, 2)
        self.haar_filter = nn.Parameter(haar_init.repeat(c1, 1, 1, 1))

        # ---------- 1×1 压缩卷积 ----------
        self.conv = nn.Conv2d(c1 * 4, c2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.conv2d(x, self.haar_filter, stride=2, groups=self.c1)
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)

    # ---------- 工具方法 ----------
    def _get_avg_filter(self):
        C = self.c1
        filters = self.haar_filter.view(C, 4, 1, 2, 2)  # (C, 4, 1, 2, 2)
        return filters.mean(dim=0)  # (4, 1, 2, 2)

    def orthogonal_loss(self):
        """正交性约束：要求四个平均滤波器两两正交"""
        avg_f = self._get_avg_filter()          # (4, 1, 2, 2)
        filters = avg_f.squeeze(1)              # (4, 2, 2)
        filters_flat = filters.reshape(4, -1)   # (4, 4)
        norm = filters_flat.norm(dim=1, keepdim=True)
        norm = torch.where(norm > 1e-8, norm, torch.ones_like(norm))
        filters_normalized = filters_flat / norm
        inner = torch.mm(filters_normalized, filters_normalized.t())
        mask = ~torch.eye(4, device=inner.device).bool()
        loss = (inner[mask] ** 2).mean()
        return loss

    def stopband_energy_loss(self, nyquist=0.4):
        """阻带能量约束：抑制高频滤波器的阻带能量"""
        avg_f = self._get_avg_filter()          # (4, 1, 2, 2)
        hf_filters = avg_f[1:].squeeze(1)       # 取 LH, HL, HH (3, 2, 2)
        loss = 0.0
        for f in hf_filters:
            f_pad = F.pad(f, (0, 30, 0, 30))    # 零填充至 32x32
            freq = torch.fft.fft2(f_pad)
            freq_mag = torch.abs(freq)
            rows, cols = freq_mag.shape
            row_axis = torch.linspace(-1, 1, rows, device=freq_mag.device)
            col_axis = torch.linspace(-1, 1, cols, device=freq_mag.device)
            Y, X = torch.meshgrid(row_axis, col_axis, indexing='ij')
            radius = torch.sqrt(X**2 + Y**2)
            stop_mask = (radius > nyquist).float()
            stop_energy = (freq_mag * stop_mask).sum()
            total_energy = freq_mag.sum() + 1e-8
            loss += stop_energy / total_energy
        return loss / 3.0

    def regularization_loss(self):
        """供外部训练器自动收集的辅助损失（无参调用）"""
        loss = 0.0
        if self.reg_w_orth > 0:
            loss += self.reg_w_orth * self.orthogonal_loss()
        if self.reg_w_stop > 0:
            loss += self.reg_w_stop * self.stopband_energy_loss(self.reg_nyquist)
        return loss

# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 模拟 YOLO11n P4 特征 (C=128, 40x40)
    x = torch.randn(2, 128, 40, 40).to(device)

    # 标准下采样对比
    conv_standard = nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False).to(device)
    hwd = HWD(128, 256).to(device)

    out_conv = conv_standard(x)
    out_hwd = hwd(x)

    print("=== 输出形状对比 ===")
    print(f"标准 Conv: {out_conv.shape}")  # [2, 256, 20, 20]
    print(f"HWD      : {out_hwd.shape}")   # [2, 256, 20, 20]

    # 参数量对比
    def count_params(module):
        return sum(p.numel() for p in module.parameters())

    print("\n=== 参数量对比 ===")
    print(f"标准 Conv: {count_params(conv_standard):,}")
    print(f"HWD      : {count_params(hwd):,}")
    print(f"HWD 参数量仅为标准卷积的 {count_params(hwd)/count_params(conv_standard)*100:.1f}%")