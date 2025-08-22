from .dilate_attention_3d import MDLABlock
import torch
import torch.nn as nn
from mmdet3d.models.builder import BACKBONES
from mmcv.runner import BaseModule


@BACKBONES.register_module()
class MDLA3D(BaseModule):
    def __init__(self,
                 dim,
                 num_blocks,
                 num_heads, 
                 kernel_size, 
                 dilation,
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop=0.,  # mlp drop
                 attn_drop=0.,  # attn drop
                 drop_path=0.,  # drop path
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 cpe_per_block=False):
        super(MDLA3D, self).__init__()
        drop_path = [x.item() for x in torch.linspace(0, drop_path, num_blocks)]
        self.blocks = nn.ModuleList([
            MDLABlock(dim=dim * (i+1), num_heads=num_heads,
                      kernel_size=kernel_size, dilation=dilation,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                      qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                      norm_layer=norm_layer, act_layer=act_layer, cpe_per_block=cpe_per_block)
            for i in range(num_blocks)])
        
        self.downsamples = nn.ModuleList(
            [nn.Conv3d(in_channels=dim * (i+1),
                       out_channels=dim * (i+2),
                       kernel_size=(3, 3, 3),
                       stride=2,
                       padding=1) for i in range(num_blocks-1)]
        )
    
    def forward(self, x):
        res = []
        for ii, ds in enumerate(self.downsamples):
            x = self.blocks[ii](x)
            res.append(x)
            x = ds(x)
        x = self.blocks[-1](x)
        res.append(x)
        return res

