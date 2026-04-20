import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import os

# 论文：SMFANet: A Lightweight Self-Modulation Feature Aggregation Network for Efficient Image Super-Resolution (ECCV 2024)
# 论文地址：https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Ren_The_Ninth_NTIRE_2024_Efficient_Super-Resolution_Challenge_Report_CVPRW_2024_paper.pdf

# ---------- 原始模块（保持不变） ----------
class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x

class PCFN(nn.Module):
    def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)
        self.conv_0 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.conv_1 = nn.Conv2d(p_dim, p_dim, 3, 1, 1)
        self.act = nn.GELU()
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        self.p_dim = p_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        if self.training:
            x = self.act(self.conv_0(x))
            x1, x2 = torch.split(x, [self.p_dim, self.hidden_dim - self.p_dim], dim=1)
            x1 = self.act(self.conv_1(x1))
            x = self.conv_2(torch.cat([x1, x2], dim=1))
        else:
            x = self.act(self.conv_0(x))
            x[:, :self.p_dim, :, :] = self.act(self.conv_1(x[:, :self.p_dim, :, :]))
            x = self.conv_2(x)
        return x

class SMFA(nn.Module):
    def __init__(self, dim=36):
        super(SMFA, self).__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.lde = DMlp(dim, 2)
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.gelu = nn.GELU()
        self.down_scale = 8
        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, f):
        _, _, h, w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h, w),
                                mode='nearest')
        y_d = self.lde(y)
        return self.linear_2(x_l + y_d)

class FMB(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()
        self.smfa = SMFA(dim)
        self.pcfn = PCFN(dim, ffn_scale)

    def forward(self, x):
        x = self.smfa(F.normalize(x)) + x
        x = self.pcfn(F.normalize(x)) + x
        return x

# ---------- 新增：完整SMFANet网络 ----------
class SMFANet(nn.Module):
    def __init__(self, dim=36, num_blocks=4, scale=2, input_channels=3):
        """
        Args:
            dim (int): 特征通道数
            num_blocks (int): FMB模块数量
            scale (int): 超分辨率放大倍数
            input_channels (int): 输入图像通道数（RGB为3）
        """
        super(SMFANet, self).__init__()
        self.scale = scale

        # 浅层特征提取
        self.conv_first = nn.Conv2d(input_channels, dim, 3, 1, 1)

        # 多个FMB模块
        self.fmbs = nn.Sequential(*[FMB(dim) for _ in range(num_blocks)])

        # 上采样模块（PixelShuffle）
        self.upsample = nn.Sequential(
            nn.Conv2d(dim, dim * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale)
        )

        # 输出卷积
        self.conv_last = nn.Conv2d(dim, input_channels, 3, 1, 1)

    def forward(self, x):
        # 假设输入x为低分辨率图像，形状 (B, C, H, W)
        feat = self.conv_first(x)      # 浅层特征
        feat = self.fmbs(feat)         # 深层特征
        out = self.upsample(feat)       # 上采样到高分辨率
        out = self.conv_last(out)       # 重建RGB
        return out

# ---------- 图像处理流程 ----------
def process_image(input_path, output_path, model, device):
    # 读取图像
    img = Image.open(input_path).convert('RGB')
    # 转换为Tensor [0,1]
    transform = transforms.ToTensor()
    lr_tensor = transform(img).unsqueeze(0).to(device)  # (1, C, H, W)

    # 模型推理
    with torch.no_grad():
        sr_tensor = model(lr_tensor)  # (1, C, H*scale, W*scale)

    # 后处理：裁剪到有效范围并转换为图像
    sr_tensor = sr_tensor.squeeze(0).cpu().clamp(0, 1)
    sr_img = transforms.ToPILImage()(sr_tensor)

    # 保存图像
    sr_img.save(output_path)
    print(f"超分辨率图像已保存至: {output_path}")

if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型（放大2倍，使用4个FMB，特征通道36）
    model = SMFANet(dim=36, num_blocks=4, scale=2).to(device)
    model.eval()  # 测试模式

    # 测试输入输出（用随机张量验证）
    dummy_input = torch.randn(1, 3, 64, 64).to(device)
    dummy_output = model(dummy_input)
    print(f"输入尺寸: {dummy_input.shape} -> 输出尺寸: {dummy_output.shape}")

    # 处理真实图片（请替换为你的图片路径）
    input_image_path =  "D:/Study/PostGraduate/YOLO/datasets/SSDD/1.png"     # 替换为输入图片路径
    output_image_path = "D:/Study/PostGraduate/YOLO/datasets/SSDD/2.png"    # 输出图片路径

    if os.path.exists(input_image_path):
        process_image(input_image_path, output_image_path, model, device)
    else:
        print(f"请将输入图片放置为 {input_image_path}，或修改路径。")