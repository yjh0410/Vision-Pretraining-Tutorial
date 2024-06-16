import torch
import torch.nn as nn


# ------------ 定义 Patch Embedding 层 ------------
class PatchEmbed(nn.Module):
    def __init__(self,
                 in_chans    : int = 3,
                 embed_dim   : int = 768,
                 kernel_size : int = 16,
                 padding     : int = 0,
                 stride      : int = 16,
                 ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

bs, c, img_h, img_w = 2, 3, 128, 256
p = 8
embed_dim = 384

# 随机生成一组图像
img = torch.randn(bs, c, img_h, img_w)

# 定义层
layer = PatchEmbed(in_chans=c, embed_dim=embed_dim, kernel_size=p, padding=0, stride=p)

# patch embedding处理
out = layer(img)
print("输如的shape：", img.shape)
print("输出的shape：", out.shape)

# 转换为序列
## [B, C, H, W] -> [B, C, N] -> [B, N, C]
out = out.flatten(2).permute(0, 2, 1).contiguous()
print("序列的shape：", out.shape)