import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


# 随机生成一张图像
B, C_in, img_h, img_w = 4, 5, 32, 32
x = torch.randn(B, C_in, img_h, img_w)

# --------------------------- 卷积 ---------------------------
# 定义一层卷积
C_out = 32
conv_layer = nn.Conv2d(in_channels=C_in, out_channels=C_out,
                         kernel_size=[5, 3], padding=[0, 0], stride=[2, 1], dilation=[1, 1],
                         bias=True)

# 打印线性层的权重矩阵的shape，即数据的维度
print("权重矩阵的shape: ", conv_layer.weight.shape)

# 打印线性层的偏置向量的shape
print("偏置矩阵的shape: ", conv_layer.bias.shape)

# 线性映射
z = conv_layer(x)

# 打印输入和输出的shape
print("输入图像的shape: ", x.shape)
print("输出图像的shape: ", z.shape)

# 定义Sigmoid激活函数
activation = nn.Sigmoid()
conv_layer = nn.Sequential(
    nn.Conv2d(in_channels=C_in, out_channels=C_out,
                         kernel_size=[5, 3], padding=[0, 0], stride=[2, 1], dilation=[1, 1],
                         bias=True),
    nn.Sigmoid()
)

# 定义卷积核
weight = torch.randn(C_out, C_in, 5, 3)
bias   = torch.randn(C_out)
z2 = F.conv2d(x, weight=weight, bias=bias, stride=[2, 1], padding=[0, 0])

# 打印输入和输出的shape
print("输入图像的shape: ", x.shape)
print("输出图像的shape: ", z2.shape)

# 读取图片
img = cv2.imread("tom_jerry.jpeg")
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
print(img_gray.shape)

# 高斯平滑
img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0.0)

# 可视化
cv2.imshow("original image", img_gray)
cv2.waitKey(0)

# 将图像转换为torch.Tensor类型，并调整shape：[H, W] -> [B, C, H, W]，其中，B=1，C=1
img_tensor = torch.from_numpy(img_gray)[None, None].float()

# 定义边缘提取算子
kernel = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]).float()
kernel = kernel[None, None]

# 卷积计算，提取边缘信息
result = F.conv2d(img_tensor, kernel, padding=0, stride=1)

# 阈值化处理
result = (result.abs() >= 127).float() * 255.

# 转换numpy格式，便于使用cv2可视化
result = result[0, 0].numpy().astype(np.uint8)

# 可视化
cv2.imshow("contouring", result)
cv2.waitKey(0)


# --------------------------- 池化 ---------------------------
# 随机生成一张图像
B, C_in, img_h, img_w = 4, 16, 32, 32
x = torch.randn(B, C_in, img_h, img_w)

# 定义一层最大池化层
maxpooling = nn.MaxPool2d(kernel_size=(2, 2), padding=(0, 0), stride=(2, 2))

# 最大池化操作
z = maxpooling(x)

# 打印输入和输出的shape
print("输入图像的shape: ", x.shape)
print("输出图像的shape: ", z.shape)

# 定义一层平均池化层
avgpooling = nn.AvgPool2d(kernel_size=(2, 2), padding=(0, 0), stride=(2, 2))

# 平均池化操作
z = avgpooling(x)

# 打印输入和输出的shape
print("输入图像的shape: ", x.shape)
print("输出图像的shape: ", z.shape)


# --------------------------- 简单的三层卷积神经网络 ---------------------------
class ConvNet(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        ) 
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ) 
        self.layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ) 

    def forward(self, x):
        x = self.layer_1(x)
        print("Layer-1 output shape: ", x.shape)

        x = self.layer_2(x)
        print("Layer-2 output shape: ", x.shape)

        x = self.layer_3(x)
        print("Layer-3 output shape: ", x.shape)

        return x

# 随机生成一张图像
B, C_in, img_h, img_w = 4, 16, 32, 32
x = torch.randn(B, C_in, img_h, img_w)

# 定义一层最大池化层
C_out = 256
model = ConvNet(in_dim=C_in, out_dim=C_out)

# 最大池化操作
y = model(x)
