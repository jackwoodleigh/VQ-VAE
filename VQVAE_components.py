import math

import torch
from torch import nn
from Blocks import ResLayerBuilder
from torch.nn import functional as F

class Block(nn.Module):

    def __init__(self, block_type, d_model, block_structure, scale_structure, block_multiplier, in_channels=None, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # will always be reassigned for encoder and for decoders with optional input convs
        previous = d_model * block_multiplier[-1]

        # must define the input for encoder and output for decoder
        assert (block_type == -1 and in_channels is not None) or (block_type == 1 and out_channels is not None)

        self.input_conv = nn.Identity()
        # defines a convolution structure for all input channels
        if in_channels is not None:
            assert len(in_channels) > 1
            self.input_conv = nn.Sequential()

            for i in range(len(in_channels) - 1):
                self.input_conv.append(nn.Conv2d(in_channels[i], in_channels[i + 1], kernel_size=3, padding=1))

            previous = in_channels[-1]

        self.output_conv = nn.Identity()
        # defines a convolution structure for all output channels
        if out_channels is not None:
            assert len(out_channels) > 1
            self.output_conv = nn.Sequential()

            for i in range(len(out_channels) - 1):
                self.output_conv.append(nn.Conv2d(out_channels[i], out_channels[i + 1], kernel_size=3, padding=1))

        # defines all resnet sub-blocks
        self.layers = ResLayerBuilder(block_type, previous, d_model, block_structure, scale_structure, block_multiplier)

    def forward(self, x):
        x = self.input_conv(x)
        for module in self.layers:
            x = module(x)
        return self.output_conv(x)

 # https://github.com/lucidrains/vector-quantize-pytorch
 # https://github.com/kakaobrain/rq-vae-transformer/tree/main

class Quantizer(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim, decay=0.9):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embeddings_dim = embeddings_dim
        self.decay = decay
        self.eps = 1e-5

        self.register_buffer("codebook", torch.randn(embeddings_dim, num_embeddings))
        self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer("ema_input_sum_per_embedding", self.codebook.clone())

        nn.init.uniform_(self.codebook, -1/self.num_embeddings, 1/self.num_embeddings)

        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def forward(self, inputs):
        # flatten input
        flatten = inputs.view(-1, self.embeddings_dim)

        # compute distance from encoder outputs to codebook vectors
        '''l2_input = torch.sum(flatten.pow(2), dim=1, keepdim=True)
        input_dot_codebook = flatten @ self.codebook
        l2_codebook = torch.sum(self.codebook.pow(2), dim=0, keepdim=True)

        distances = l2_input - 2 * input_dot_codebook + l2_codebook'''
        distances = (
                torch.sum(flatten ** 2, dim=1, keepdim=True)
                - 2 * torch.matmul(flatten, self.codebook)
                + torch.sum(self.codebook ** 2, dim=0, keepdim=True)
        )

        distances = self.distance(flatten)

        # finding the nearest codebook vector indices and one hot it
        nearest_codebook_indices = torch.argmin(distances, dim=-1).unsqueeze(1)
        nearest_codebook_indices_one_hot = F.one_hot(nearest_codebook_indices, self.num_embeddings).squeeze(1).type(
            inputs.dtype)

        # updating codebook vectors with EMA
        if self.training:
            # total number of uses for each embedding
            nearest_codebook_indices_one_hot_sum = nearest_codebook_indices_one_hot.sum(dim=0)

            # sum of encoder output vectors assigned to each codebook vector
            input_sum_per_embedding = flatten.transpose(0, 1) @ nearest_codebook_indices_one_hot

            # updating ema of cluster size
            self.ema_cluster_size.data.mul_(self.decay).add_((1 - self.decay) * nearest_codebook_indices_one_hot_sum)
            self.ema_input_sum_per_embedding.data.mul_(self.decay).add_((1 - self.decay) * input_sum_per_embedding)

            # total size of all clusters
            n = self.ema_cluster_size.sum()

            # adds small number eps to all clusters for no division by 0
            # normalizes cluster+eps by new total sum that includes eps
            norm_cluster_size = ((self.ema_cluster_size + self.eps) / (n + self.num_embeddings * self.eps)) * n

            # moving codebook clusters by mean
            self.codebook.data = self.ema_input_sum_per_embedding / norm_cluster_size.unsqueeze(0)


        avg_probs = nearest_codebook_indices_one_hot.float().mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # use the one hot encodings for nearest vectors to get codebook vectors
        quantized = (nearest_codebook_indices_one_hot @ self.codebook.t()).view(inputs.shape)

        commitment_loss = torch.mean((quantized.detach() - inputs).pow(2))

        # straight through estimator
        quantized = inputs + (quantized - inputs).detach()

        return quantized, commitment_loss, perplexity


    def distance(self, flatten):
        inputs_norm_sq = flatten.pow(2.).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = self.codebook.pow(2.).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            flatten,
            self.codebook,
            alpha=-2.0,
        )
        return distances