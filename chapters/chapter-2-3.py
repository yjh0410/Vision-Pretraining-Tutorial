import torch
import torch.nn as nn


# 随即生成数据：[B, C, H, W]
bs, c, img_h, img_w = 2, 5, 13, 7
x = torch.randn(bs, c, img_h, img_w)

# -------- 使用 nn.AdaptiveAvgPool2d 类来实现全局平均池化操作 --------
avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

y = avgpool(x)
print(y.shape)  # [B, C, 1, 1]
print(y[:, :, 0, 0])

# -------- 使用 torch.mean 函数来实现全局平均池化操作 --------
# [B, C, H, W] -> [B, C, HW]
z = x.flatten(2)
z = torch.mean(z, dim=2)
print(z)
