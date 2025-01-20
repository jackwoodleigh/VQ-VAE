import math

import torch
from torch import nn
from Blocks import res_layer_builder, ResBlock, MultiHeadSelfAttention
from torch.nn import functional as F

from scipy.cluster.vq import kmeans2
from sklearn.cluster import MiniBatchKMeans

class Encoder(nn.Module):
    def __init__(self, d_model, in_channels, mid_channels, out_channels, res_structure, res_multiplier, scale_structure):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            ResBlock(in_channels=32, out_channels=d_model * res_multiplier[0])
        )

        self.layers = res_layer_builder(
            block_type=-1,
            previous_channels=d_model * res_multiplier[0],
            d_model=d_model,
            block_structure=res_structure,
            block_multiplier=res_multiplier,
            scale_structure=scale_structure
        )

        self.mid_layers = nn.Sequential(
            ResBlock(in_channels=d_model * res_multiplier[-1], out_channels=8),
            ResBlock(in_channels=8, out_channels=8),
            ResBlock(in_channels=8, out_channels=8),
            ResBlock(in_channels=8, out_channels=8),
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
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
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
        )

        self.mid_layers = nn.Sequential(
            ResBlock(in_channels=mid_channels, out_channels=mid_channels),
            ResBlock(in_channels=mid_channels, out_channels=mid_channels),
            ResBlock(in_channels=mid_channels, out_channels=mid_channels),
            ResBlock(in_channels=mid_channels, out_channels=d_model * res_multiplier[-1]),
        )

        self.layers = res_layer_builder(
            block_type=1,
            previous_channels=d_model * res_multiplier[-1],
            d_model=d_model,
            block_structure=res_structure,
            block_multiplier=res_multiplier,
            scale_structure=scale_structure
        )

        output_in_channels = d_model * res_multiplier[0]
        self.output_conv = nn.Sequential(
            ResBlock(in_channels=output_in_channels, out_channels=64),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 16, kernel_size=3, padding=1),
            nn.GroupNorm(8, 16),
            nn.SiLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1)
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
        self.kld_scale = 10.0
        self.dead_code_threshold = 0.25

        self.codebook = nn.Embedding(self.num_embeddings, self.embeddings_dim)
        self.codebook.weight.data.uniform_(-100.0 / self.num_embeddings, 100.0 / self.num_embeddings)

        self.register_buffer("cluster_sizes", torch.zeros(self.num_embeddings))
        self.register_buffer("codebook_value_ema", self.codebook.weight.clone())
        self.register_buffer('data_initialized', torch.zeros(1))

    def forward(self, inputs, beta, replace_dead_codes=False, ema=False):
        dtype = inputs.dtype
        # flatten input
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        flatten = inputs.view(-1, self.embeddings_dim)

        # compute distance from encoder outputs to codebook vectors
        d = self.distance(flatten)

        # finding the nearest codebook vector indices
        nearest_codebook_indices = torch.argmin(d, dim=1)
        # one hot indices (num_input_vec, num_embeddings)
        nearest_codebook_indices_one_hot = F.one_hot(nearest_codebook_indices, self.num_embeddings).to(dtype=dtype)

        # use the one hot encodings for nearest vectors to get codebook vectors
        quantized = torch.matmul(nearest_codebook_indices_one_hot, self.codebook.weight.to(dtype=dtype)).view(inputs.shape)

        loss = None
        if self.training:

            # used for ema and dead codebook entry resets
            self.codebook_tracker_updates(flatten, nearest_codebook_indices_one_hot)

            # commitment loss
            loss = beta * torch.mean((quantized - inputs.detach()) ** 2)

            if ema:
                self.codebook_ema_updates()

            else:
                # adding codebook loss if no ema
                loss += torch.mean((quantized.detach() - inputs) ** 2)
                loss *= self.kld_scale

            # replacing unused codebook entries
            if replace_dead_codes:
                self.replace_dead_codes(flatten)

        # straight through estimator
        quantized = inputs + (quantized.detach() - inputs.detach())

        # perplexity
        e_mean = torch.mean(nearest_codebook_indices_one_hot, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # return shape
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return loss, quantized, perplexity

    def distance(self, flatten):
        d = flatten.pow(2).sum(1, keepdim=True) - 2 * flatten @ self.codebook.weight.t() + self.codebook.weight.pow(2).sum(1, keepdim=True).t()
        #d = torch.sum(flatten ** 2, dim=1, keepdim=True) + torch.sum(self.codebook.weight ** 2, dim=1) - 2 * torch.matmul(flatten, self.codebook.weight.t())
        return d

    # https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py
    def codebook_tracker_updates(self, flatten, one_hot):
        # finding how much each entry is used
        one_hot_sum = one_hot.sum(0)

        # finding the values of closest encoder values for each codebook entry
        flatten_sum = flatten.t() @ one_hot

        self.cluster_sizes.data.mul_(self.decay).add_(one_hot_sum, alpha=1 - self.decay)
        self.codebook_value_ema.data.mul_(self.decay).add_(flatten_sum.t(), alpha=1 - self.decay)

    # https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py
    def codebook_ema_updates(self):
        n = self.cluster_sizes.data.sum()

        # laplace smoothing
        cluster_size = (self.cluster_sizes.data + self.eps) / (n + self.num_embeddings * self.eps) * n

        codebook_ema_normalized = self.codebook_value_ema.data / cluster_size.unsqueeze(1)

        self.codebook.weight = nn.Parameter(codebook_ema_normalized)

    def replace_dead_codes(self, flatten):
        dead_codes_mask = self.cluster_sizes < self.dead_code_threshold
        new_centroids = dead_codes_mask.sum().item()

        if new_centroids == 0:
            return

        # todo: will run into errors if num of flatten (num, embedding_dim) is smaller than num of codes in codebook
        '''
        centroids, labels = kmeans2(flatten.to(torch.float32).cpu().detach().numpy(), new_centroids, minit='++')

        new_codes = torch.tensor(centroids, dtype=self.codebook.weight.dtype, device=self.codebook.weight.device)
        noise_scale = 0.001
        noise = torch.randn_like(new_codes) * noise_scale
        new_codes_with_noise = new_codes + noise
        '''

        new_codes = self.new_localized_cluster_dispersing_codes(new_centroids)

        with torch.no_grad():
            self.codebook.weight.data[dead_codes_mask] = new_codes

        self.cluster_sizes[dead_codes_mask] = self.dead_code_threshold
        self.codebook_value_ema[dead_codes_mask] = new_codes * self.dead_code_threshold


    # creates new random codes within the proximity of the most used codes
    # the average distance between codes in the code book is used as proximity/magnitude
    def new_localized_cluster_dispersing_codes(self, n):
        top_sizes, top_indices = torch.topk(self.cluster_sizes.data, k=n, largest=True)
        top_indices = self.size_proportional_new_cluster_assignments(top_sizes, top_indices)

        # finds the average distance between clusters
        distances = self.distance(self.codebook.weight)
        # find the distance to vectors in close-proximity
        closest_values, closest_indices = torch.topk(distances, k=10, dim=1, largest=False)
        avg_distances = torch.mean(closest_values, dim=1)[top_indices].unsqueeze(1)

        print(f"Number of new clusters: {n}, largest cluster: {top_sizes[0]}, avg distance: {avg_distances.mean()}")

        # creates random direction tensor with magnitude of 1
        direction = 2 * torch.randn(n, self.embeddings_dim, device=self.codebook.weight.device, dtype=self.codebook.weight.dtype) - 1
        norm_direction = torch.nn.functional.normalize(direction, p=2, dim=1)

        # random magnitude from 0.25 to 1. don't want new clusters to overlap with old
        random_magnitude = 0.65 * torch.randn(n, device=self.codebook.weight.device, dtype=self.codebook.weight.dtype).unsqueeze(1) + 0.1

        # offset used to create new clusters near large clusters
        offset = random_magnitude * avg_distances * norm_direction

        # essentially defines a high dimensional ring around large clusters where new clusters will spawn
        new_codes = self.codebook.weight[top_indices].detach() + offset

        return new_codes

    def size_proportional_new_cluster_assignments(self, usage_counts, top_indices):
        size = len(top_indices)

        norm = usage_counts.clone().detach() / sum(usage_counts.detach())
        counts = torch.floor(norm * size).long()

        i = 0
        while counts.sum() < size:
            counts[i] += 1
            i += 1

        indices = torch.repeat_interleave(torch.arange(len(counts), dtype=usage_counts.dtype, device=usage_counts.device), counts).long()
        indices = top_indices[indices]
        return indices
