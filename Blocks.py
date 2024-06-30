
import torch
from torch import nn
from torch.nn import functional as F


def res_layer_builder(block_type, previous_channels, d_model, block_structure, block_multiplier, scale_structure):
    layers = []
    scale_structure = [block_type*x for x in scale_structure]

    # is decoder
    if block_type == 1:
        # follows chain of blocks in reverse
        struct = reversed(list(enumerate(block_structure)))

    # is encoder
    else:
        # follows chain of blocks
        struct = enumerate(block_structure)

    # builds structure
    for i, count in struct:
        temp = []
        current_channels = d_model * block_multiplier[i]

        # add layers resblocks
        for block in range(count):
            layers.append(ResBlock(previous_channels, current_channels))
            previous_channels = current_channels

        # place up sample at the start
        if scale_structure[i] == 1:
            temp.insert(0, ScaleBlock(current_channels, scale_structure[i]))

        # place down sample at the end
        elif scale_structure[i] == -1:
            temp.append(ScaleBlock(current_channels, scale_structure[i]))

        temp.append(MultiHeadSelfAttention(8, current_channels))

        layers.extend(temp)

    return nn.Sequential(*layers)

class ScaleBlock(nn.Module):
    def __init__(self, channels, scale_type=0):
        super().__init__()
        # -1 for down sample
        if scale_type == -1:
            self.scale_block = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

        # 1 for up sample
        elif scale_type == 1:
            self.scale_block = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

        # otherwise no sample
        else:
            self.scale_block = nn.Identity()

    def forward(self, x):
        return self.scale_block(x)

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, n_heads, channels, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_projection = nn.Linear(channels, 3*channels, bias=in_proj_bias)
        self.out_projection = nn.Linear(channels, channels, bias=out_proj_bias)
        self.group_norm = nn.GroupNorm(32, channels)

        self.channels = channels
        self.n_heads = n_heads
        self.d_heads = channels // n_heads

        self.ReZero_weight = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = x
        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1).view(B, H*W, C)

        qkv = self.in_projection(x)

        qkv = qkv.view(B, H*W, self.n_heads, C*3 // self.n_heads)

        q, k, v = qkv.chunk(3, dim=-1)

        x = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        x = x.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
        output = self.out_projection(x)
        output = output.view(B, C, H, W)

        return residual + self.ReZero_weight * output

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        # residual channels
        if in_channels == out_channels:
            self.res_out_conv = nn.Identity()
        else:
            self.res_out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.ReZero_weight = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = x
        x = self.block1(x)
        x = self.block2(x)

        # https://arxiv.org/pdf/2003.04887
        x = self.res_out_conv(residual) + self.ReZero_weight * x

        return x

    def forward2(self, x):
        residual = x
        x = self.block1(x)
        x = self.sample_block(x)
        x = self.block2(x)
        x = self.block3(x)

        residual = self.res_out_conv(self.res_sample_block(residual))
        return residual + self.ReZero_weight * x





