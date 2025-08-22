import torch
import torch.nn as nn
import torch.nn.functional as F
from projects.occ_plugin.utils import Unfold3D


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    "Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff input sizes that may have diff dim.
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

    def extra_repr(self) -> str:
        return f'drop_prob={self.drop_prob}'


class DilateAttention3D(nn.Module):
    "3D implementation of Dilate-attention"
    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** - 0.5
        self.kernel_size = kernel_size
        self.unfold = Unfold3D(kernel_size, dilation, dilation*(kernel_size-1)//2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self,q,k,v):
        # B, C//num_dilation, D, H, W
        B, d, D, H, W = q.shape
        q = q.reshape([B, d//self.head_dim, self.head_dim, 1, D, H, W]).permute(0, 1, 4, 5, 6, 3, 2)  # B, h, D, H, W, 1, d
        k = self.unfold(k).reshape([B, d//self.head_dim, self.head_dim, self.kernel_size**3, D, H, W]).permute(0, 1, 4, 5, 6, 2, 3)  # B, h, D, H, W, d, k^3
        attn = (q @ k) * self.scale  # B, h, D, H, W, 1, k^3
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape([B, d//self.head_dim, self.head_dim, self.kernel_size**3, D, H, W]).permute(0, 1, 4, 5, 6, 3, 2)  # B, h, D, H, W, k^3, d
        x = (attn @ v).transpose(1, 5).reshape(B, D, H, W, d)
        return x


class MultiDilatelocalAttention3D(nn.Module):
    "3D implementation of multi-dilate local attention"

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
        self.qkv = nn.Conv3d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [DilateAttention3D(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, D, H, W, C = x.shape
        x = x.permute(0, 4, 1, 2, 3)  # B, C, D, H, W
        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C//self.num_dilation, D, H, W).permute(2, 1, 0, 3, 4, 5, 6)
        # num_dilation, 3, B, C//num_dilation, D, H, W
        x = x.reshape(B, self.num_dilation, C//self.num_dilation, D, H, W).permute(1, 0, 3, 4, 5, 2)
        # num_dilation, B, D, H, W, C//num_dilation
        res = torch.zeros_like(x)
        for i in range(self.num_dilation):
            res[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])  # B, D, H, W, C//num_dilation
        x = res.permute(1, 2, 3, 4, 0, 5).reshape(B, D, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MDLABlock(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None,  
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, 
                 kernel_size=3, 
                 dilation=[1, 2, 3],
                 cpe_per_block=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiDilatelocalAttention3D(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            kernel_size=kernel_size, dilation=dilation)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.cpe_per_block = cpe_per_block
        if self.cpe_per_block:
            self.pos_embed = nn.Conv3d(dim, dim, (3, 3, 3), padding=1, groups=dim)


    def forward(self, x):
        if self.cpe_per_block:
            x = x + self.pos_embed(x)  # [B, C, H, W, D]
        x = x.permute(0, 4, 2, 3, 1)  # [B, D, H, W, C]
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # [B, D, H, W, C]
        x = x.permute(0, 4, 2, 3, 1)  # [B, C, H, W, D]
        return x
