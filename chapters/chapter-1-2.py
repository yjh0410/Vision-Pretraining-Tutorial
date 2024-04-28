import torch
import torch.nn as nn


# 生成一批数据
B, C_in, C_out = 7, 256, 1024
x = torch.randn(B, C_in)

# --------------------------- 单层感知机 ---------------------------
# 定义一层线性层
linear_layer = nn.Linear(in_features=C_in, out_features=C_out, bias=True)

# 打印线性层的权重矩阵的shape，即数据的维度
print("权重矩阵的shape: ", linear_layer.weight.shape)

# 打印线性层的偏置向量的shape
print("偏置矩阵的shape: ", linear_layer.bias.shape)

# 线性映射
z = linear_layer(x)

# 打印线性输出的shape
print(z.shape)

print(linear_layer.weight.requires_grad)
print(linear_layer.bias.requires_grad)

# 定义Sigmoid激活函数
activation = nn.Sigmoid()

# 非线性激活输出
y = activation(z)

# 打印非线性输出的shape
print(y.shape)

import torch.nn.functional as F
y2 = F.sigmoid(z)
diff = y - y2
print(diff.sum().item())

# 标准的PyTorch规范的神经网络模型搭建
class SingleLayerPerceptron(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.layer = nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
        self.act   = nn.Sigmoid()

    def forward(self, x):
        z = self.layer(x)
        y = self.act(z)

        return y

# 基于上方的单层感知机类，来构建一个模型
model = SingleLayerPerceptron(in_dim=C_in, out_dim=C_out)

# 模型前向推理
y = model(x)

# 打印输出的shape
print(y.shape)

# --------------------------- 多层感知机 ---------------------------
print("================= 多层感知机 =================")
# 生成一批数据
B, C_in, C_out = 3, 256, 1
x = torch.randn(B, C_in)

# 标准的PyTorch规范的神经网络模型搭建
class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_dim, inter_dim, out_dim) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            # 第一层感知机
            nn.Linear(in_features=in_dim, out_features=inter_dim, bias=True),
            nn.Sigmoid(),
            # 第二层感知机
            nn.Linear(in_features=inter_dim, out_features=inter_dim, bias=True),
            nn.Sigmoid(),
            # 第三层感知机
            nn.Linear(in_features=inter_dim, out_features=out_dim, bias=True),
            nn.Sigmoid(),
            )

    def forward(self, x):
        y = self.layer(x)

        return y

# 基于上方的多层感知机类，来构建一个模型
model = MultiLayerPerceptron(in_dim=C_in, inter_dim=2048, out_dim=C_out)

# 模型前向推理
y = model(x)

# 打印输出的shape
print(y.shape)

# 定义标签
target = torch.randint(low=0, high=2, size=[B, 1]).float()
print("标签: ", target)

# 定义二元交叉熵损失函数
criterion = nn.BCELoss(reduction='none')

# 计算二元交叉熵损失
loss = criterion(y, target)
print("二元交叉熵损失：", loss)
print("二元交叉熵损失的shape：", loss.shape)

import torch.nn.functional as F
loss = criterion(y, target)
loss2 = F.binary_cross_entropy(y, target, reduction='none')
diff = loss - loss2
print(diff)

# 定义二元交叉熵损失函数
criterion2 = nn.BCEWithLogitsLoss(reduction='none')
y = torch.tensor([-2, 3, 4]).float()
target = torch.tensor([1, 0, 1]).float()
loss1 = criterion(F.sigmoid(y), target)
print(loss1)
loss2 = criterion2(y, target)
print(loss2)
loss3 = F.binary_cross_entropy_with_logits(y, target, reduction='none')
print(loss3)

# --------------------------- 多分类任务的多层感知机 ---------------------------
print("================= 多分类任务的多层感知机 =================")
# 生成一批数据
B, C_in, C_out = 3, 256, 4
x = torch.randn(B, C_in)

# 标准的PyTorch规范的神经网络模型搭建
class MLP(nn.Module):
    def __init__(self, in_dim, inter_dim, out_dim) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            # 第一层感知机
            nn.Linear(in_features=in_dim, out_features=inter_dim, bias=True),
            nn.Sigmoid(),
            # 第二层感知机
            nn.Linear(in_features=inter_dim, out_features=inter_dim, bias=True),
            nn.Sigmoid(),
            # 第三层感知机
            nn.Linear(in_features=inter_dim, out_features=out_dim, bias=True),
            nn.Softmax(dim=1),
            )

    def forward(self, x):
        y = self.layer(x)

        return y

# 基于上方的多层感知机类，来构建一个模型
model = MLP(in_dim=C_in, inter_dim=2048, out_dim=C_out)

# 模型前向推理
y = model(x)

# 打印输出的shape
print(y.shape)

# 定义标签
target = torch.randint(low=0, high=C_out, size=[B,]).long()
print("标签: ", target)

# 定义交叉熵损失函数
criterion = nn.CrossEntropyLoss(reduction='none')

# 计算交叉熵损失
loss = criterion(y, target)
print("交叉熵损失：", loss)
print("交叉熵损失的shape：", loss.shape)

loss2 = F.cross_entropy(y, target, reduction='none')
print("交叉熵损失：", loss2)
print("交叉熵损失的shape：", loss2.shape)

out = torch.tensor([[123, 80000, -112, 30], [8, 4, 2, 3]]).float()
print("Softmax处理结果：", F.softmax(out, dim=1))
