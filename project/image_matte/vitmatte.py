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
from typing import Tuple, List
import todos

import pdb

# --------------------------------------------------------------------------------------------------
# xxxx_debug
# // partition into non-overlapping windows with padding if needed
# // example:
# // a:   768   64   64    1
# // w:    14
# // res: 768   14   14    25
# // used in sam
# GGML_API struct ggml_tensor * ggml_win_part(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   w);
def window_partition(x, window_size:int) -> Tuple[torch.Tensor, int, int]:
    # [1, 42, 50, 384] --> [12, 14, 14, 384]
    # window_size = 14
    B, H, W, C = x.shape
    # todos.debug.output_var("window_partition-1", x)
    assert window_size == 14
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    # windows.size() -- [54, 14, 14, 384]
    # (Hp, Wp) -- (126, 84)

    # todos.debug.output_var("window_partition-2", windows)
    # print("-" * 80)
    # tensor [window_partition-1] size: [1, 42, 50, 384], min: -6.848723, max: 10.042427, mean: 0.045886
    # tensor [window_partition-2] size: [12, 14, 14, 384], min: -6.848723, max: 10.042427, mean: 0.040969

    return windows, Hp, Wp

# xxxx_debug
# // reverse of ggml_win_part
# // used in sam
# GGML_API struct ggml_tensor * ggml_win_unpart(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   w0,
#         int                   h0,
#         int                   w);
def window_unpartition(windows, window_size:int, pad_hw:Tuple[int, int], hw:Tuple[int, int]):
    # tensor [windows] size: [20, 14, 14, 384], min: -7.176075, max: 1.416744, mean: -0.005811
    # window_size = 14
    # pad_hw = (70, 56)
    # hw = (64, 43)

    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    x = x[:, :H, :W, :].contiguous() # x.size() -- [1, 120, 80, 384]

    # tensor [x] size: [1, 64, 43, 384], min: -7.057968, max: 1.416744, mean: -0.006378

    return x

# xxxx_debug
# GGML_API struct ggml_tensor * ggml_get_rel_pos(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   qh,
#         int                   kh);
def get_rel_pos(rel_pos, q_size:int, k_size:int):
    # rel_pos.size() -- [27, 64]
    # q_size = 14
    # k_size = 14

    max_rel_dist = int(2 * max(q_size, k_size) - 1) # ==> 27, 79, 119
    # Interpolate rel pos.
    rel_pos_resized = F.interpolate(
        rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
        size=max_rel_dist,
        mode="linear",
    )
    rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]

# xxxx_debug
# GGML_API struct ggml_tensor * ggml_add_rel_pos(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * pw,
#         struct ggml_tensor  * ph);
def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size:Tuple[int, int], k_size:Tuple[int, int]):
    # tensor [attn] size: [120, 196, 196], min: -8.433104, max: 16.265987, mean: 1.114269
    # tensor [q] size: [120, 196, 64], min: -7.24711, max: 7.226545, mean: 0.014577
    # tensor [rel_pos_h] size: [27, 64], min: -0.012905, max: 0.012103, mean: -7.8e-05
    # tensor [rel_pos_w] size: [27, 64], min: -0.013695, max: 0.012973, mean: -9.8e-05

    # q_size = (14, 14)
    # k_size = (14, 14)

    q_h, q_w = q_size
    k_h, k_w = k_size
    # if not (q_h == 14 and q_w == 14 and k_h == 14 and k_w == 14):
    #     (Pdb) k_size -- (64, 43)

    Rh = get_rel_pos(rel_pos_h, q_h, k_h)
    Rw = get_rel_pos(rel_pos_w, q_w, k_w)

    B, _, dim = q.shape # [120, 196, 64]
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh) # [120, 14, 14, 14]
    # tensor [r_q] size: [6, 64, 43, 64], min: -3.554034, max: 4.482345, mean: 0.045117
    # tensor [Rh] size: [64, 64, 64], min: -0.172288, max: 0.167465, mean: -0.000355
    # tensor [rel_h] size: [6, 64, 43, 64], min: -3.037493, max: 5.377393, mean: 0.004754

    # r_q.size() -- [b=120, h=14, w=14, (c=64)]
    # Rh.size() -- [h=14, k=14, (c=64)]
    # ==> [b=120, h=14, w=14, k=14]

    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw) # [120, 14, 14, 14]
    # tensor [r_q] size: [120, 14, 14, 64], min: -5.109172, max: 5.019894, mean: 0.006196
    # tensor [Rw] size: [14, 14, 64], min: -0.137916, max: 0.143502, mean: 0.000514
    # tensor [rel_w] size: [120, 14, 14, 14], min: -3.59078, max: 5.638031, mean: -0.029699
    # r_q = [b, h, w, (c)]
    # Rw [w, k, (c)]
    # ==> [b, h, w, k]

    # xxxx_debug
    # print("B, q_h, q_w, k_h, k_w -- ", B, q_h, q_w, k_h, k_w)
    # todos.debug.output_var("attn", attn)
    # todos.debug.output_var("rel_h", rel_h)
    # todos.debug.output_var("rel_w", rel_w)
    # print("-" * 80)
    # B, q_h, q_w, k_h, k_w --  6 64 43 64 43
    # tensor [attn] size: [6, 2752, 2752], min: -16.951553, max: 20.740995, mean: 1.040415
    # tensor [rel_h] size: [6, 64, 43, 64], min: -3.037493, max: 5.377393, mean: 0.004754
    # tensor [rel_w] size: [6, 64, 43, 43], min: -1.91948, max: 4.111428, mean: 0.006052

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)
    # [120, 196, 196] -> [120, 14, 14, 14, 14] + ... -> [120, 196, 196]
    # attn == ggml_add_rel_pos(attn, rel_w, rel_h) ?

    return attn # attn.size() -- [120, 196, 196]

def get_abs_pos(abs_pos, hw:Tuple[int, int]):
    # tensor [abs_pos] size: [1, 197, 384], min: -0.161103, max: 0.160729, mean: 1.1e-05
    # hw = (64, 43)
    h, w = hw
    abs_pos = abs_pos[:, 1:] # ==> [1, 196, 384]
    xy_num = abs_pos.shape[1] # ===> 196
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num
    assert size == 14

    # abs_pos.reshape(1, size, size, -1).size() -- [1, 14, 14, 384]
    new_abs_pos = F.interpolate(
        abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2), #[] -> [1, 384, 14, 14]
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    )
    return new_abs_pos.permute(0, 2, 3, 1) # [1, 384, 64, 43] --> [1, 64, 43, 384]

class PatchEmbed(nn.Module):
    def __init__(self, in_chans=4, embed_dim=384, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features,
        hidden_features=None,
        out_features=None,
    ):
        super().__init__()
        assert in_features == 384
        assert hidden_features == 1536
        assert out_features == None

        out_features = out_features or in_features
        assert out_features == in_features

        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

# xxxx_debug
class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""
    def __init__(self, dim = 384, num_heads=6, input_size=(14, 14)):
        super().__init__()
        assert dim == 384 and num_heads == 6 
        # and input_size[0] == 14 and input_size[1] == 14, (32, 32)

        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert head_dim == 64

        self.scale = head_dim**-0.5 # 0.125

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        # initialize relative positional embeddings
        self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim)) # size() -- [27, 64]
        self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim)) # size() -- [27, 64]

    def forward(self, x):
        B, H, W, C = x.shape # [20, 14, 14, 384]
        # qkv with shape (3, B, nHead, H * W, C)
        # if not (H == 14 and W == 14):
        #     x.size() -- torch.Size([1, 64, 43, 384])

        # xxxx_debug
        # self.qkv(x).size() -- [20, 14, 14, 1152]
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # [20, 14, 14, 1152] -> [20, 196, 3, 6, 64] -> [3, 20, 6, 196, 64]
        # ggml: [20, 14, 14, 1152] -> [20, 196, 1152] -> [20, 196, 18, 64]-> [20, 196, 3x6, 64]
        # -> 3x[20, 196, 6, 64] --> 3x[20, 6, 196, 64] 

        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0) # torch.size() -- [120, 196, 64]
        # [3, 120, 196, 64] --> [120, 196, 64] x 3
        attn = (q * self.scale) @ k.transpose(-2, -1) # size() -- [120, 196, 196]
        # xxxx_debug        
        attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        # size() -- [120, 196, 196]

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        # [120, 196, 64] -> [20, 6, 14, 14, 64] -> [20, 14, 14, 6, 64] -> [20, 14, 14, 384]
        # ggml: [120, 196, 64] -> [20, 6, 196, 64] -> [20, 196, 6, 64] -> [20, 196, 384] -> [20, 14, 14, 384]

        x = self.proj(x)
        # tensor [x] size: [20, 14, 14, 384], min: -477.986969, max: 1045.842041, mean: -1.808324
        return x


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-6
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        # tensor [x] size: [1, 384, 64, 43], min: -0.201075, max: 0.193223, mean: 0.000799
        u = x.mean(1, keepdim=True)
        # tensor [u] size: [1, 1, 64, 43], min: -0.015526, max: 0.015471, mean: 0.000799
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        # tensor [x] size: [1, 384, 64, 43], min: -9.696175, max: 10.70018, mean: -0.0
        # tensor [self.weight] size: [384], min: -0.682143, max: 0.702993, mean: -0.003777
        # tensor [self.bias] size: [384], min: -0.178839, max: 0.146375, mean: 0.007882
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ResBottleneckBlock(nn.Module):
    """
    The standard bottleneck residual block without the last activation layer.
    It contains 3 conv layers with kernels 1x1, 3x3, 1x1.
    """
    def __init__(self, in_channels, out_channels, bottleneck_channels,
    ):
        super().__init__()
        assert in_channels == 384 and out_channels == 384 and bottleneck_channels == 192

        # self.in_channels = in_channels
        # self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = LayerNorm(bottleneck_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3,
            padding=1,
            bias=False,
        )
        self.norm2 = LayerNorm(bottleneck_channels)
        self.act2 = nn.GELU() # ggml_gelu()

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


class BlockForNormalWindow(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(self,
        dim = 384,
        num_heads = 6,
        window_size=14, # 14 or 0 
    ):
        super().__init__()
        assert dim == 384 and num_heads == 6 and window_size == 14

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, input_size= (window_size, window_size))

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * 4))

        self.window_size = window_size


    def forward(self, x):
        shortcut = x
        x = self.norm1(x)

        # Window partition
        # Support torch.jit.script
        pad_h:int = 0
        pad_w:int = 0
        H:int = 0
        W:int = 0
        H, W = x.shape[1], x.shape[2]
        x, pad_h, pad_w = window_partition(x, self.window_size)

        x = self.attn(x)

        # Reverse window partition
        x = window_unpartition(x, self.window_size, (pad_h, pad_w), (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class BlockForZeroWindow(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(self,
        dim = 384,
        num_heads = 6,
        input_size=(32, 32),
    ):
        super().__init__()
        assert dim == 384 and num_heads == 6 and input_size[0] == 32 and input_size[1] == 32

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, input_size=input_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim * 4)

        # Use a residual block with bottleneck channel as dim // 2
        self.residual = ResBottleneckBlock(
            in_channels=dim,
            out_channels=dim,
            bottleneck_channels=dim // 2,
        )


    def forward(self, x):
        shortcut = x
        x = self.norm1(x)

        # Window partition
        x = self.attn(x)

        # Reverse window partition
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        # x.size() -- [1, 64, 43, 384]
        # x.permute(0, 3, 1, 2).size() -- [1, 384, 64, 43]

        x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # [1, 384, 64, 43] -> [1, 64, 43, 384]

        return x


class ViT(nn.Module):
    """
    This module implements Vision Transformer (ViT) backbone in :paper:`vitdet`.
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """
    def __init__(self,
        patch_size=16,
        in_chans=4,
        embed_dim=384,
        num_heads=6,
        window_size=14,
        window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],
    ):
        super().__init__()
        assert patch_size == 16 and in_chans == 4
        assert embed_dim == 384 and num_heads == 6 and window_size == 14
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # Initialize absolute positional embedding with pretrain image size.
        num_patches = (224 // patch_size) * (224 // patch_size) # 14 * 14 ==> 196
        num_positions = (num_patches + 1) # 197

        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim)) # size() -- [1, 197, 384]

        self.blocks = nn.ModuleList()

        depth = 12
        for i in range(depth): # depth -- 12
            if i in window_block_indexes: # [0, 1, 3, 4, 6, 7, 9, 10]
                block = BlockForNormalWindow(
                    dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                )
            else:
                block = BlockForZeroWindow(
                    dim=embed_dim,
                    num_heads=num_heads,
                    input_size=(512 // patch_size, 512 // patch_size), # (32, 32)
                )

            self.blocks.append(block)


    def forward(self, x):
        # tensor [x] size: [1, 4, 1024, 688], min: -2.117904, max: 2.610589, mean: 0.374034
        x = self.patch_embed(x) # B, H, W, C
        x = x + get_abs_pos(self.pos_embed, (x.shape[1], x.shape[2])) # H, W ?

        for blk in self.blocks:
            x = blk(x)

        # tensor [x] size: [1, 64, 43, 384], min: -10.419563, max: 32.650574, mean: 0.029471
        return x.permute(0, 3, 1, 2) # [1, 64, 43, 384] -> [1, 384, 64, 43] -- (B, C, H, W)


# --------------------------------------------------------------------------------------------------
class BasicConv3x3(nn.Module):
    """
    Basic convolution layers including: Conv3x3, BatchNorm2d, ReLU layers.
    """
    def __init__(self, in_chans, out_chans, stride=2, padding=1):
        super().__init__()
        assert padding == 1

        self.conv = nn.Conv2d(in_chans, out_chans, 3, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class ConvStream(nn.Module):
    """
    Simple ConvStream containing a series of basic conv3x3 layers to extract detail features.
    """
    def __init__(self, in_chans = 4, out_chans = [48, 96, 192]):
        super().__init__()
        self.convs = nn.ModuleList()
        
        self.conv_chans = out_chans.copy()
        self.conv_chans.insert(0, in_chans)
        # self.conv_chans -- [4, 48, 96, 192]
        for i in range(len(self.conv_chans)-1):
            in_chan_ = self.conv_chans[i]
            out_chan_ = self.conv_chans[i+1]
            self.convs.append(BasicConv3x3(in_chan_, out_chan_)) # [4, 48], [48, 96], [96, 192]
    
    def forward(self, x) -> List[torch.Tensor]:
        # tensor [x] size: [1, 4, 1024, 688], min: -2.117904, max: 2.610589, mean: 0.374034
        out_list: List[torch.Tensor] = [x]
        for i, m in enumerate(self.convs): # 3
            x = m(x)
            out_list.append(x)

        # todos.debug.output_var("out_list", out_list)
        # out_list is list: len = 4
        #     tensor [item] size: [1, 4, 1024, 688], min: -2.117904, max: 2.610589, mean: 0.374034
        #     tensor [item] size: [1, 48, 512, 344], min: 0.0, max: 14.644271, mean: 0.46483
        #     tensor [item] size: [1, 96, 256, 172], min: 0.0, max: 19.618439, mean: 0.449103
        #     tensor [item] size: [1, 192, 128, 86], min: 0.0, max: 17.098135, mean: 0.477254
        return out_list # out_dict

class FusionBlock(nn.Module):
    """
    Simple fusion block to fuse feature from ConvStream and Plain Vision Transformer.
    """
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.conv = BasicConv3x3(in_chans, out_chans, stride=1, padding=1)

    def forward(self, x, D):
        up = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        out = torch.cat([D, up], dim=1)
        out = self.conv(out)

        return out    

class MattingHead(nn.Module):
    """
    Simple Matting Head, containing only conv3x3 and conv1x1 layers.
    """
    def __init__(self, in_chans = 32, mid_chans = 16):
        super().__init__()
        assert mid_chans == 16

        self.matting_convs = nn.Sequential(
            nn.Conv2d(in_chans, mid_chans, 3, 1, 1),
            nn.BatchNorm2d(mid_chans),
            nn.ReLU(),
            nn.Conv2d(mid_chans, 1, 1, 1, 0)
        )

    def forward(self, x):
        return self.matting_convs(x)

class DetailCapture(nn.Module):
    """
    Simple and Lightweight Detail Capture Module for ViT Matting.
    """
    def __init__(self, in_chans = 384, img_chans=4, 
        convstream_out = [48, 96, 192], fusion_out = [256, 128, 64, 32]):
        super().__init__()
        assert in_chans == 384 and img_chans == 4

        assert len(fusion_out) == len(convstream_out) + 1

        self.convstream = ConvStream(in_chans = img_chans)
        self.conv_chans = self.convstream.conv_chans # [4, 48, 96, 192]

        self.fusion_blks = nn.ModuleList()
        self.fus_channs = fusion_out.copy()
        self.fus_channs.insert(0, in_chans) # [384, (256, 128, 64, 32)]

        for i in range(len(self.fus_channs)-1):
            self.fusion_blks.append(
                FusionBlock(in_chans = self.fus_channs[i] + self.conv_chans[-(i+1)], out_chans = self.fus_channs[i+1])
            )
        # [576, 256], [352, 128], [176, 64], [68, 32]
        self.matting_head = MattingHead(in_chans = fusion_out[-1]) # fusion_out[-1] -- 32

    def forward(self, features, images):
        detail_features = self.convstream(images)
        for i, m in enumerate(self.fusion_blks): # len(self.fusion_blks) -- 4 -- 4
            features = m(features, detail_features[len(self.fusion_blks)-i-1]) # D3, D2, D1, D0
        
        return torch.sigmoid(self.matting_head(features))


# --------------------------------------------------------------------------------------------------
class ViTMatte(nn.Module):
    def __init__(self):
        super().__init__()
        self.MAX_H = 1024
        self.MAX_W = 2048
        self.MAX_TIMES = 16
        # GPU -- 1024x1024, 2.1G, 90ms
        # GPU -- 1024x2048, 7.8G, 310ms

        self.backbone = ViT()
        self.decoder = DetailCapture()
        self.normal = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), inplace=False)

        self.load_weights()
        # pdb.set_trace()
        # from ggml_engine import create_network
        # create_network(self)

    def load_weights(self, model_path="models/vitmatte.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        self.load_state_dict(torch.load(checkpoint))


    def forward(self, x):
        '''
            x is Bx4xHxW tensor, alpha channel of x is trimap
        '''
        B, C, H, W = x.size() # [1, 4, 672, 992]

        pad_h = (self.MAX_TIMES - (H % self.MAX_TIMES)) % self.MAX_TIMES
        pad_w = (self.MAX_TIMES - (W % self.MAX_TIMES)) % self.MAX_TIMES
        x = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')

        # normalize
        images = x[:, 0:3, :, :]
        trimap = x[:, 3:4, :, :]
        images = self.normal(images)

        images = torch.cat((images, trimap), dim=1)
        features = self.backbone(images)            # size() -- [1, 384, 42, 62]
        mask = self.decoder(features, images)       # size() -- [1, 1, 672, 992]

        output = torch.cat((x[:, 0:3, :, :], mask), dim=1)

        return output[:, :, 0:H, 0:W]

