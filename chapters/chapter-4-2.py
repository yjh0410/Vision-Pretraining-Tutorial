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
seq = out.flatten(2).permute(0, 2, 1).contiguous()
print("序列的shape：", seq.shape)


# ------------ 自注意力操作 ------------
## QKV映射层
qkv_proj = nn.Linear(embed_dim, embed_dim*3, bias=False)
qkv = qkv_proj(seq)
q, k, v = torch.chunk(qkv, 3, dim=-1)
print("输入的Q的shape：", q.shape)
print("输入的K的shape：", k.shape)
print("输入的V的shape：", v.shape)

## 多头自注意力操作
bs, N = q.shape[:2]
num_heads = 8
head_dim = embed_dim // num_heads
scale = head_dim ** -0.5
## [B, N, C] -> [B, N, H, C_h] -> [B, H, N, C_h]
q = q.view(bs, N, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
k = k.view(bs, N, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
v = v.view(bs, N, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
print("用于多头输入的Q的shape：", q.shape)
print("用于多头输入的K的shape：", k.shape)
print("用于多头输入的V的shape：", v.shape)

## [B, H, Nq, C_h] × [B, H, C_h, Nk] = [B, H, Nq, Nk]
attn = q * scale @ k.transpose(-1, -2)
attn = attn.softmax(dim=-1)
print("注意力矩阵或相似度矩阵的shape：", attn.shape)
x = attn @ v
print("注意力输出的shape：", x.shape)

## 输出层映射
proj = nn.Linear(embed_dim, embed_dim)
x = x.permute(0, 2, 1, 3).contiguous().view(bs, N, -1)
x = proj(x)
print("线性输出的shape：", x.shape)


# ------------ 定义 自注意力层 ------------
class Attention(nn.Module):
    def __init__(self,
                 dim       :int,
                 qkv_bias  :bool  = False,
                 num_heads :int   = 8,
                 dropout   :float = 0.
                 ):
        super().__init__()
        # --------------- Basic parameters ---------------
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # --------------- Network parameters ---------------
        self.qkv_proj = nn.Linear(dim, dim*3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        bs, N, _ = x.shape
        # ----------------- Input proj -----------------
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # ----------------- Multi-head Attn -----------------
        ## [B, N, C] -> [B, N, H, C_h] -> [B, H, N, C_h]
        q = q.view(bs, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.view(bs, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.view(bs, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        ## [B, H, Nq, C_h] X [B, H, C_h, Nk] = [B, H, Nq, Nk]
        attn = q * self.scale @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v # [B, H, Nq, C_h]

        # ----------------- Output -----------------
        x = x.permute(0, 2, 1, 3).contiguous().view(bs, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

attn_layer = Attention(embed_dim, num_heads=8)
attn_out = attn_layer(seq)
print("注意力曾的输出的shape：", attn_out.shape)


# ------------ 定义 前馈传播网络（FFN）------------
class FeedFroward(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 mlp_dim: int,
                 act: nn.GELU,
                 dropout: float = 0.0,
                 ) -> None:
        super().__init__()
        self.fc1   = nn.Linear(embedding_dim, mlp_dim)
        self.drop1 = nn.Dropout(dropout)
        self.fc2   = nn.Linear(mlp_dim, embedding_dim)
        self.drop2 = nn.Dropout(dropout)
        self.act   = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
