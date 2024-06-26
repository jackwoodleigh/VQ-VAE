
import torch
from torch import nn
from torch.nn import functional as F

def ResLayerBuilder(block_type, previous_channels, d_model, block_structure, scale_structure, block_multiplier):
    layers = nn.ModuleList()
    scale_structure = [block_type * x for x in scale_structure]

    # is decoder
    if block_type == 1:
        # follows chain of blocks in reverse
        struct = reversed(list(enumerate(block_structure)))

    # is encoder
    else:
        # follows chain of blocks
        struct = enumerate(block_structure)

    for i, count in struct:
        current_channels = d_model * block_multiplier[i]
        for block in range(count):
            scale = 0

            # for last block in sub layer attach the scale type
            if block == count - 1:
                scale = scale_structure[i]

            layers.append(ResBlock(previous_channels, current_channels, scale_type=scale))
            previous_channels = current_channels

    return layers

class SampleBlock(nn.Module):
    def __init__(self, channels, scale_type=0):
        super().__init__()
        # -1 for down sample
        if scale_type == -1:
            self.scale_block = nn.Conv2d(channels, channels, kernel_size=2, stride=2, padding=0)

        # 1 for up sample
        elif scale_type == 1:
            self.scale_block = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2, padding=0)

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
    def __init__(self, in_channels, out_channels, scale_type=0, dropout_rate=0.1):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
        )

        #self.sample_block = SampleBlock(in_channels, scale_type)

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels)
        )
        self.block3 = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        # residual channels
        if in_channels == out_channels:
            self.res_out_conv = nn.Identity()
        else:
            self.res_out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # residual sample type
        self.res_sample_block = SampleBlock(out_channels, scale_type)

        self.ReZero_weight = nn.Parameter(torch.zeros(1))

        self.sample_block = SampleBlock(out_channels, scale_type)

        self.attention = MultiHeadSelfAttention(8, out_channels)

    def forward(self, x):
        residual = x
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # https://arxiv.org/pdf/2003.04887
        x = self.res_out_conv(residual) + self.ReZero_weight * x

        x = self.attention(x)
        x = self.sample_block(x)

        return x

    def forward2(self, x):
        residual = x
        x = self.block1(x)
        x = self.sample_block(x)
        x = self.block2(x)
        x = self.block3(x)

        residual = self.res_out_conv(self.res_sample_block(residual))
        return residual + self.ReZero_weight * x




