import torch
import torch.nn as nn
import torch.nn.functional as F

class InfSA(nn.Module):
    def __init__(self, channels, num_steps=2, gamma=0.5, eps=1e-8):
        super().__init__()
        self.channels = channels
        self.num_steps = num_steps
        self.gamma = gamma
        self.eps = eps

        self.proj_q = nn.Conv2d(channels, channels, 1, bias=False)
        self.proj_k = nn.Conv2d(channels, channels, 1, bias=False)
        self.proj_v = nn.Conv2d(channels, channels, 1, bias=False)
        self.out_proj = nn.Conv2d(channels, channels, 1)

        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q = self.proj_q(x).flatten(2).transpose(1, 2)   # (B, N, C)
        k = self.proj_k(x).flatten(2).transpose(1, 2)
        v = self.proj_v(x).flatten(2).transpose(1, 2)

        # 使用更稳定的归一化：RMSNorm 或 LayerNorm
        q = F.normalize(q, p=2, dim=-1, eps=self.eps)
        k = F.normalize(k, p=2, dim=-1, eps=self.eps)
        v = F.normalize(v, p=2, dim=-1, eps=self.eps)

        # 初始化 h
        h = v

        # 迭代更新
        for _ in range(self.num_steps):
            kv = torch.bmm(k.transpose(1, 2), h)  # (B, C, C)
            # 对 kv 进行归一化，防止数值爆炸
            kv = kv / (torch.norm(kv, dim=1, keepdim=True) + self.eps)
            agg = torch.bmm(q, kv)               # (B, N, C)
            h = (1 - self.gamma) * h + self.gamma * agg
            # 可选：对 h 进行 clip
            h = torch.clamp(h, -10, 10)

        h = self.norm(h)
        out = h.transpose(1, 2).reshape(B, C, H, W)
        out = self.out_proj(out)
        return out

if __name__ == '__main__':
    # 单元测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InfSA(channels=64, num_steps=3, gamma=0.8).to(device)
    x = torch.randn(2, 64, 32, 32).to(device)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    # 梯度测试
    loss = out.mean()
    loss.backward()
    print("Backward pass completed.")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} has gradient.")
        else:
            print(f"Warning: {name} has no gradient.")