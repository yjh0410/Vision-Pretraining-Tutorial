import torch

def random_masking(x, mask_ratio=0.75):
    # 输入的 x 是图像序列，即一组token
    # mask ratio便是被随机遮掩的token数量的比例
    B, N, C = x.shape
    len_keep = int(N * (1 - mask_ratio))  # 要保留的token的数量

    # 随机初始化与输入序列相等长度的噪声
    noise = torch.rand(B, N, device=x.device)

    # 根据噪声的数值来做排序，由于噪声是随机的，因此这里的排序也就等于随机操作了
    # 后续将会根据这个排序来保留一部分token，
    # 丢弃一部分token，丢弃的token就等效于被遮掩的图像patch
    ids_shuffle = torch.argsort(noise, dim=1)

    # 记录排序后的每个token在原始序列中的位置，
    # 以便我们后续去把保留的token放置在原位，同时填充mask token
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    print(noise)
    print(ids_shuffle)
    print(ids_restore)

    # 保留的token的位置索引
    ids_keep = ids_shuffle[:, :len_keep]

    # 利用torch 的gather函数来根据 ids_keep 变量中的索引得到全部的被保留的token
    # 这些token将组成新的输入序列
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))

    # generate the binary mask: 0 is keep, 1 is remove
    # 同时生成一组mask，以便后续在计算损失时，避免被保留的token所预测的像素参与损失计算
    mask = torch.ones([B, N], device=x.device)
    mask[:, :len_keep] = 0

    # 根据 ids_restore 变量中的位置索引，将准备好的 mask 转换成与原始序列对应的格式，
    # 其中的 0 表示被遮掩的token标记，1 表示被保留的token标记
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore
    
B, N, C = 1, 6, 8
x = torch.randn(B, N, C)
x_masked, mask, ids_restore = random_masking(x, mask_ratio=0.5)