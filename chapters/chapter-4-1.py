import torch
num_seq = 6
num_dim = 16
X = [torch.randn(num_dim) for _ in range(num_seq)]

# -------- 循环：计算第2个向量与整个序列的相似度 --------
print("---- 采用循环来计算相似度 ----")
def softmax_for_list(array):
    array_exp = [torch.exp(array[i]) for i in range(len(array))]
    div = sum(array_exp)
    return [array_exp[i] / div for i in range(len(array))]

## 第一步：计算点积
s_2 = [torch.sum(X[2]*X[i]) for i in range(num_seq)]
print("点积计算结果：", s_2)

## 第二步：计算相似度
s_2 = softmax_for_list(s_2)
print("相似度：", s_2)

# -------- 矩阵乘法：计算第2个向量与整个序列的相似度 --------
print("---- 采用矩阵乘法来计算相似度 ----")
def softmax_for_tensor(array):
    array_exp = torch.exp(array)
    div = torch.sum(array_exp, dim=1, keepdim=True)
    return array_exp / div

X = torch.stack(X)            # [N, C]
x_2 = X[2].unsqueeze(0)       # [1, C]
s_2 = torch.matmul(x_2, X.T)  # [1, N]
print("点积计算结果：", s_2)

## 第二步：计算相似度
s_2 = softmax_for_tensor(s_2)
print("相似度：", s_2)

