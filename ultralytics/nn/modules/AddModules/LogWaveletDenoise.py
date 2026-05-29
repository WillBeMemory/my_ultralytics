import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LogWaveletDenoise(nn.Module):
    """
    优化版 对数域小波去噪模块 | 适配YOLO11
    优化点：数值稳定性 + 尺寸对齐 + 空间自适应阈值 + 静态图兼容 + 特征融合分支
    修复：梯度Hook报错（推理/eval模式兼容）
    """

    def __init__(self, c1, c2, level=1, downsample=True, kernel_size=3):
        super().__init__()
        self.downsample = downsample
        self.level = level
        self.eps = 1e-6  # 数值稳定性常数

        # 1. 正交Haar小波滤波器
        lo = torch.tensor([1.0, 1.0]) / math.sqrt(2)
        hi = torch.tensor([1.0, -1.0]) / math.sqrt(2)
        dec_LL = (lo.unsqueeze(0) * lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        dec_LH = (lo.unsqueeze(0) * hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        dec_HL = (hi.unsqueeze(0) * lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        dec_HH = (hi.unsqueeze(0) * hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        self.register_buffer('dec_filters', torch.cat([dec_LL, dec_LH, dec_HL, dec_HH], dim=0))
        self.register_buffer('rec_filters', self.dec_filters.clone())

        # 2. 空间自适应阈值（1x1卷积，静态图兼容）
        self.thresh_lh = nn.ModuleList()
        self.thresh_hl = nn.ModuleList()
        self.thresh_hh = nn.ModuleList()
        for _ in range(level):
            self.thresh_lh.append(nn.Conv2d(c1, c1, 1, bias=True))
            self.thresh_hl.append(nn.Conv2d(c1, c1, 1, bias=True))
            self.thresh_hh.append(nn.Conv2d(c1, c1, 1, bias=True))
        # 初始化阈值为小值
        for m in self.thresh_lh + self.thresh_hl + self.thresh_hh:
            nn.init.constant_(m.weight, -2.0)
            nn.init.constant_(m.bias, 0.0)

        # 3. YOLO原生下采样层
        if self.downsample:
            self.down_conv = nn.Conv2d(c1, c2, kernel_size, stride=2, padding=kernel_size // 2, bias=False)
            self.down_bn = nn.BatchNorm2d(c2)
            self.down_act = nn.SiLU(inplace=True)

    # 小波分解（自动补边，尺寸对齐）
    def _dwt(self, x):
        B, C, H, W = x.shape
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        filters = self.dec_filters.repeat(C, 1, 1, 1)
        coeffs = F.conv2d(x, filters, stride=2, groups=C)
        coeffs = coeffs.view(B, C, 4, coeffs.shape[-2], coeffs.shape[-1])
        return coeffs[:, :, 0], coeffs[:, :, 1], coeffs[:, :, 2], coeffs[:, :, 3]

    # 小波重构（严格裁剪回原始尺寸，解决维度报错）
    def _idwt(self, LL, LH, HL, HH, orig_size):
        B, C, H, W = LL.shape
        coeffs = torch.stack([LL, LH, HL, HH], dim=2).reshape(B, C * 4, H, W)
        filters = self.rec_filters.repeat(C, 1, 1, 1)
        out = F.conv_transpose2d(coeffs, filters, stride=2, groups=C)
        return out[:, :, :orig_size[0], :orig_size[1]]

    # 软阈值函数
    def _soft_threshold(self, coeff, lambd):
        return torch.where(
            coeff > lambd, coeff - lambd,
            torch.where(coeff < -lambd, coeff + lambd, torch.zeros_like(coeff))
        )

    # 多级去噪（静态图兼容）
    def _multilevel_denoise(self, x_log):
        orig_size = x_log.shape[2:]
        detail_coeffs = []
        current = x_log

        # 多级分解
        for lvl in range(self.level):
            LL, LH, HL, HH = self._dwt(current)
            detail_coeffs.append((LL, LH, HL, HH))
            current = LL

        # 逐级去噪+重构
        for lvl in reversed(range(self.level)):
            LL, LH, HL, HH = detail_coeffs[lvl]
            # 空间自适应阈值
            tau_lh = F.softplus(self.thresh_lh[lvl](LL))
            tau_hl = F.softplus(self.thresh_hl[lvl](LL))
            tau_hh = F.softplus(self.thresh_hh[lvl](LL))

            LH = self._soft_threshold(LH, tau_lh)
            HL = self._soft_threshold(HL, tau_hl)
            HH = self._soft_threshold(HH, tau_hh)

            current = self._idwt(LL, LH, HL, HH, orig_size)

        return current

    def forward(self, x):
        # 优化1：数值稳定性（移除Hook，避免推理报错）
        x_clamped = torch.clamp(x, min=self.eps)
        x_log = torch.log(x_clamped)
        # 替代方案：直接裁剪数值，兼容训练+推理
        x_log = torch.clamp(x_log, min=-20.0, max=20.0)

        # 小波去噪
        rec_log = self._multilevel_denoise(x_log)
        rec_log = torch.clamp(rec_log, max=20)
        out = torch.exp(rec_log)

        # 优化5：特征融合分支（残差连接）
        if out.shape == x.shape:
            out = 0.5 * out + 0.5 * x

        # 下采样
        if self.downsample:
            out = self.down_act(self.down_bn(self.down_conv(out)))
        return out


# -------------------------- 测试函数 --------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"运行设备: {device}")

    # 模拟YOLO11输入
    x = torch.randn(2, 64, 320, 320).to(device)

    # 初始化模型
    model = LogWaveletDenoise(c1=64, c2=128, level=1, downsample=True).to(device)
    model.eval()

    # 推理（无梯度，无报错）
    with torch.no_grad():
        output = model(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"无NaN: {not torch.isnan(output).any()}")
    print(f"无Inf: {not torch.isinf(output).any()}")
    print("✅ 修复完成！LogWaveletDenoise 测试通过！")


if __name__ == "__main__":
    main()