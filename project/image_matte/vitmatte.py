"""Create model."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright 2023 Dell(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, Wed 28 Jun 2023 05:46:59 PM CST
# ***
# ************************************************************************************/
#

import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as T
from typing import Tuple, Dict, List
import pdb

# --------------------------------------------------------------------------------------------------
def window_partition(x, window_size:int) -> Tuple[torch.Tensor, int, int]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    # window_size = 14
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    # windows.size() -- [54, 14, 14, 384]
    # (Hp, Wp) -- (126, 84)

    return windows, Hp, Wp


def window_unpartition(windows, window_size:int, pad_hw:Tuple[int, int], hw:Tuple[int, int]):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    # windows.size(), window_size, pad_hw, hw
    # ([54, 14, 14, 384], 14, (126, 84), (120, 80))
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()

    # x.size() -- [1, 120, 80, 384]

    return x


def get_rel_pos(q_size:int, k_size:int, rel_pos):
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    # q_size = 14
    # k_size = 14
    # rel_pos.size() -- [27, 64]

    max_rel_dist = int(2 * max(q_size, k_size) - 1) # ==> 27
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist: # False
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size:Tuple[int, int], k_size:Tuple[int, int]):
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    # attn.size() -- [324, 196, 196]

    return attn


def get_abs_pos(abs_pos, has_cls_token:bool, hw:Tuple[int, int]):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    # abs_pos.size() -- [1, 196, 384]
    # has_cls_token = True
    # hw -- (120, 80)

    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1] # 196
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w: # ==> True
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0), in_chans=3, embed_dim=768
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super(PatchEmbed, self).__init__()
        # in_chans = 4
        # embed_dim = 384

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)

        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super(Mlp, self).__init__()
        # in_features = 384
        # hidden_features = 1536
        # out_features = None
        # norm_layer = None
        # bias = True
        # drop = 0.0

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super(Attention, self).__init__()
        # dim = 384
        # num_heads = 6
        # use_rel_pos = True
        # input_size = (14, 14)

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos: # True
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

            # if not rel_pos_zero_init: # ==> False
            #     trunc_normal_(self.rel_pos_h, std=0.02)
            #     trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ResBottleneckBlock(nn.Module):
    """
    The standard bottleneck residual block without the last activation layer.
    It contains 3 conv layers with kernels 1x1, 3x3, 1x1.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        act_layer=nn.GELU,
        conv_kernels=3,
        conv_paddings=1,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            norm (str or callable): normalization for all conv layers.
            act_layer (callable): activation for all conv layers.
        """
        super(ResBottleneckBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.stride = 1

        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = LayerNorm(bottleneck_channels)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            conv_kernels,
            padding=conv_paddings,
            bias=False,
        )
        self.norm2 = LayerNorm(bottleneck_channels)
        self.act2 = act_layer()

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = LayerNorm(out_channels)

        for layer in [self.norm1, self.norm2]:
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()
        # zero init last norm layer.
        self.norm3.weight.data.zero_()
        self.norm3.bias.data.zero_()

    def forward(self, x):
        out = x
        for layer in self.children():
            out = layer(out)

        out = x + out
        return out


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        use_residual_block=False,
        use_convnext_block=False,
        input_size=None,
        res_conv_kernel_size=3,
        res_conv_padding=1,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_residual_block (bool): If True, use a residual block after the MLP block.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super(Block, self).__init__()
        # dim = 384
        # num_heads = 6
        # use_rel_pos = True
        # window_size = 14
        # input_size = (32, 32)

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

        self.window_size = window_size

        self.use_residual_block = use_residual_block
        if use_residual_block: # False ?
            # Use a residual block with bottleneck channel as dim // 2
            self.residual = ResBottleneckBlock(
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 2,
                act_layer=act_layer,
                conv_kernels=3,
                conv_paddings=1,
            )
        else: # For torch.jit.script
            self.residual = nn.Identity()

        self.use_convnext_block = use_convnext_block
        if use_convnext_block: # False
            self.convnext = ConvNextBlock(dim = dim)
        else: # For torch.jit.script
            self.convnext = nn.Identity()


    def forward(self, x):
        shortcut = x
        x = self.norm1(x)

        # Window partition
        # Support torch.jit.script
        pad_h:int = 0
        pad_w:int = 0
        H:int = 0
        W:int = 0
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_h, pad_w = window_partition(x, self.window_size)

        x = self.attn(x)

        # Reverse window partition
        if self.window_size > 0: # True, window_size == 14
            x = window_unpartition(x, self.window_size, (pad_h, pad_w), (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        if self.use_residual_block: # False
            x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        if self.use_convnext_block: # False
            x = self.convnext(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return x

class ViT(nn.Module):
    """
    This module implements Vision Transformer (ViT) backbone in :paper:`vitdet`.
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        img_size=512,
        patch_size=16,
        in_chans=4,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        # use_abs_pos=True,
        use_rel_pos=True,
        # rel_pos_zero_init=True,
        window_size=14,
        window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],
        residual_block_indexes=[2, 5, 8, 11],
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        # out_feature="last_feat",
    ):
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            # rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            window_block_indexes (list): Indexes for blocks using window attention.
            residual_block_indexes (list): Indexes for blocks using conv propagation.
            pretrain_img_size (int): input image size for pretraining models.
            pretrain_use_cls_token (bool): If True, pretrainig models use class token.
            out_feature (str): name of the feature from the last block.
        """
        super(ViT, self).__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # if use_abs_pos: # True
        #     # Initialize absolute positional embedding with pretrain image size.
        #     num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
        #     num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
        #     self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        # else:
        #     self.pos_embed = None
        # ==>
        # Initialize absolute positional embedding with pretrain image size.
        num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
        num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i], # 0.0
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=True,
                window_size=window_size if i in window_block_indexes else 0,
                use_residual_block=i in residual_block_indexes,
                input_size=(img_size // patch_size, img_size // patch_size),
                res_conv_kernel_size=3,
                res_conv_padding=1,
            )
            self.blocks.append(block)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)

        if self.pos_embed is not None:
            # self.pretrain_use_cls_token -- True
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )
        for blk in self.blocks:
            x = blk(x)

        # self._out_features -- ['last_feat']
        # outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}

        # return outputs['last_feat']
        return x.permute(0, 3, 1, 2)


# --------------------------------------------------------------------------------------------------
class Basic_Conv3x3(nn.Module):
    """
    Basic convolution layers including: Conv3x3, BatchNorm2d, ReLU layers.
    """
    def __init__(
        self,
        in_chans,
        out_chans,
        stride=2,
        padding=1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, 3, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class ConvStream(nn.Module):
    """
    Simple ConvStream containing a series of basic conv3x3 layers to extract detail features.
    """
    def __init__(
        self,
        in_chans = 4,
        out_chans = [48, 96, 192],
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        
        self.conv_chans = out_chans.copy()
        self.conv_chans.insert(0, in_chans)
        
        for i in range(len(self.conv_chans)-1):
            in_chan_ = self.conv_chans[i]
            out_chan_ = self.conv_chans[i+1]
            self.convs.append(
                Basic_Conv3x3(in_chan_, out_chan_)
            )
    
    def forward(self, x) -> Dict[str, torch.Tensor]:
        out_dict = {'D0': x}

        # torch.jit.script NOT SUPPORT
        # for i in range(len(self.convs)):
        #     x = self.convs[i](x)
        #     name_ = 'D'+str(i+1)
        #     out_dict[name_] = x

        for i, m in enumerate(self.convs):
            x = m(x)
            name_ = 'D'+str(i+1)
            out_dict[name_] = x
        
        return out_dict

class Fusion_Block(nn.Module):
    """
    Simple fusion block to fuse feature from ConvStream and Plain Vision Transformer.
    """
    def __init__(
        self,
        in_chans,
        out_chans,
    ):
        super().__init__()
        self.conv = Basic_Conv3x3(in_chans, out_chans, stride=1, padding=1)

    def forward(self, x, D):
        F_up = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        out = torch.cat([D, F_up], dim=1)
        out = self.conv(out)

        return out    

class Matting_Head(nn.Module):
    """
    Simple Matting Head, containing only conv3x3 and conv1x1 layers.
    """
    def __init__(
        self,
        in_chans = 32,
        mid_chans = 16,
    ):
        super().__init__()
        self.matting_convs = nn.Sequential(
            nn.Conv2d(in_chans, mid_chans, 3, 1, 1),
            nn.BatchNorm2d(mid_chans),
            nn.ReLU(True),
            nn.Conv2d(mid_chans, 1, 1, 1, 0)
            )

    def forward(self, x):
        x = self.matting_convs(x)

        return x

class Detail_Capture(nn.Module):
    """
    Simple and Lightweight Detail Capture Module for ViT Matting.
    """
    def __init__(
        self,
        in_chans = 384,
        img_chans=4,
        convstream_out = [48, 96, 192],
        fusion_out = [256, 128, 64, 32],
    ):
        super().__init__()
        assert len(fusion_out) == len(convstream_out) + 1

        self.convstream = ConvStream(in_chans = img_chans)
        self.conv_chans = self.convstream.conv_chans

        self.fusion_blks = nn.ModuleList()
        self.fus_channs = fusion_out.copy()
        self.fus_channs.insert(0, in_chans)
        for i in range(len(self.fus_channs)-1):
            self.fusion_blks.append(
                Fusion_Block(
                    in_chans = self.fus_channs[i] + self.conv_chans[-(i+1)],
                    out_chans = self.fus_channs[i+1],
                )
            )

        self.matting_head = Matting_Head(
            in_chans = fusion_out[-1],
        )
        # (Pdb) self.fusion_blks
        # ModuleList(
        #   (0): Fusion_Block(
        #     (conv): Basic_Conv3x3(
        #       (conv): Conv2d(576, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        #       (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       (relu): ReLU(inplace=True)
        #     )
        #   )
        #   (1): Fusion_Block(
        #     (conv): Basic_Conv3x3(
        #       (conv): Conv2d(352, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        #       (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       (relu): ReLU(inplace=True)
        #     )
        #   )
        #   (2): Fusion_Block(
        #     (conv): Basic_Conv3x3(
        #       (conv): Conv2d(176, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        #       (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       (relu): ReLU(inplace=True)
        #     )
        #   )
        #   (3): Fusion_Block(
        #     (conv): Basic_Conv3x3(
        #       (conv): Conv2d(68, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        #       (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       (relu): ReLU(inplace=True)
        #     )
        #   )
        # )


    def forward(self, features, images):
        detail_features = self.convstream(images)

        # torch.jit.script NOT SUPPORT
        # for i in range(len(self.fusion_blks)): # len(self.fusion_blks) -- 4
        #     d_name_ = 'D'+str(len(self.fusion_blks)-i-1)
        #     features = self.fusion_blks[i](features, detail_features[d_name_])

        for i, m in enumerate(self.fusion_blks):
            d_name_ = 'D'+str(len(self.fusion_blks)-i-1)
            features = m(features, detail_features[d_name_])
        
        return torch.sigmoid(self.matting_head(features))


# --------------------------------------------------------------------------------------------------
class ViTMatte(nn.Module):
    def __init__(self):
        super(ViTMatte, self).__init__()
        self.MAX_H = 1024
        self.MAX_W = 1024
        self.MAX_TIMES = 32

        self.backbone = ViT()
        self.decoder = Detail_Capture()
        self.normal = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), inplace=False)

        self.load_weights()

    def load_weights(self, model_path="models/vitmatte.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        self.load_state_dict(torch.load(checkpoint))


    def forward(self, x):
        '''
            x is Bx4xHxW tensor, alpha channel of x is trimap
        '''
        # x.size() -- [1, 4, 672, 992]
        B, C, H, W = x.size()

        # normalize
        images = x[:, 0:3, :, :]
        trimap = x[:, 3:4, :, :]
        images = self.normal(images)

        images = torch.cat((images, trimap), dim=1)
        features = self.backbone(images) # features.size() -- [1, 384, 42, 62]
        mask = self.decoder(features, images)  # mask.size() -- [1, 1, 672, 992]

        return torch.cat((x[:, 0:3, :, :], mask), dim=1)

