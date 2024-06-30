import math

import torch
from torch import nn
from Blocks import res_layer_builder, ResBlock, MultiHeadSelfAttention
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, d_model, in_channels, mid_channels, out_channels, res_structure, res_multiplier, scale_structure):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_conv = nn.Sequential(
            nn.Conv2d(3, in_channels, kernel_size=3, padding=1)
        )

        self.layers = res_layer_builder(
            block_type=-1,
            previous_channels=in_channels,
            d_model=d_model,
            block_structure=res_structure,
            block_multiplier=res_multiplier,
            scale_structure=scale_structure
        )

        self.mid_layers = nn.Sequential(
            ResBlock(in_channels=d_model * res_multiplier[-1], out_channels=mid_channels),
            MultiHeadSelfAttention(8, mid_channels),
            ResBlock(in_channels=mid_channels, out_channels=mid_channels),
            MultiHeadSelfAttention(8, mid_channels)
        )

        self.output_conv = nn.Sequential(
            nn.GroupNorm(32, mid_channels),
            nn.SiLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.layers(x)
        x = self.mid_layers(x)
        x = self.output_conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, in_channels, mid_channels, out_channels, res_structure, res_multiplier, scale_structure):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        )

        self.mid_layers = nn.Sequential(
            ResBlock(in_channels=mid_channels, out_channels=mid_channels),
            MultiHeadSelfAttention(8, mid_channels),
            ResBlock(in_channels=mid_channels, out_channels=mid_channels),
            MultiHeadSelfAttention(8, mid_channels)
        )

        self.layers = res_layer_builder(
            block_type=1,
            previous_channels=mid_channels,
            d_model=d_model,
            block_structure=res_structure,
            block_multiplier=res_multiplier,
            scale_structure=scale_structure
        )

        output_in_channels = d_model * res_multiplier[0]
        self.output_conv = nn.Sequential(
            nn.GroupNorm(32, output_in_channels),
            nn.SiLU(),
            nn.Conv2d(output_in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.mid_layers(x)
        x = self.layers(x)
        x = self.output_conv(x)
        return x



# https://github.com/lucidrains/vector-quantize-pytorch
# https://github.com/kakaobrain/rq-vae-transformer/tree/main

class Quantizer(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim, decay=0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embeddings_dim = embeddings_dim
        self.decay = decay
        self.eps = 1e-6

        self.codebook = nn.Embedding(self.num_embeddings, self.embeddings_dim)
        self.codebook.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, inputs, beta):
        # flatten input
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        flatten = inputs.view(-1, self.embeddings_dim)

        # compute distance from encoder outputs to codebook vectors
        d = self.distance(flatten)

        # finding the nearest codebook vector indices and one hot it
        nearest_codebook_indices = torch.argmin(d, dim=1)
        nearest_codebook_indices_one_hot = F.one_hot(nearest_codebook_indices, self.num_embeddings).type(inputs.dtype)

        # use the one hot encodings for nearest vectors to get codebook vectors
        quantized = torch.matmul(nearest_codebook_indices_one_hot, self.codebook.weight).view(inputs.shape)

        # loss
        # I believe that this must be before straight through est. It took a lot of pain to figure that out
        loss = torch.mean((quantized.detach() - inputs) ** 2) + beta * torch.mean((quantized - inputs.detach()) ** 2)

        # straight through estimator
        quantized = inputs + (quantized.detach() - inputs.detach())

        # perplexity
        e_mean = torch.mean(nearest_codebook_indices_one_hot, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return loss, quantized, perplexity

    def distance(self, flatten):
        d = torch.sum(flatten ** 2, dim=1, keepdim=True) + torch.sum(self.codebook.weight ** 2, dim=1) - 2 * torch.matmul(flatten, self.codebook.weight.t())
        return d

