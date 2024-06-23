
import torch
from torch import nn

def ResBlockBuilder(type, d_model, block_structure, scale_structure, block_multiplier):
    layers = nn.ModuleList()
    scale_structure = [type * x for x in scale_structure]

    # decoder
    if type == 1:
        struct = reversed(list(enumerate(block_structure)))
        previous_channels = d_model * block_multiplier[-1] ** len(block_multiplier)

    #encoder
    else:
        struct = enumerate(block_structure)
        previous_channels = d_model

    for i, count in struct:
        current_channels = d_model * (block_multiplier[i] ** (i + 1))
        for block in range(count):
            scale = 0
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


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_type=0, dropout_rate=0.1):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
        )

        self.sample_block = SampleBlock(in_channels, scale_type)

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
            self.res_out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # residual sample type
        self.res_sample_block = SampleBlock(in_channels, scale_type)

    def forward(self, x):
        residual = x
        x = self.block1(x)
        x = self.sample_block(x)
        x = self.block2(x)
        x = self.block3(x)

        return x + self.res_out_conv(self.res_sample_block(residual))




