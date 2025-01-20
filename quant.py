
import torch
from torch import nn
from torch.nn import functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.in_proj = nn.Conv2d(1, embedding_dim, 1)
        self.out_proj = nn.Conv2d(embedding_dim, 1, 1)

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.xavier_uniform_(self.embeddings.weight)

        self.kld_scale = 10.0
    def __call__(self, x):
        B, C, H, W = x.size()

        x = self.in_proj(x)
        x = x.permute(0, 2, 3, 1)
        flatten = x.reshape(-1, self.embedding_dim)

        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embeddings.weight.t()
                + self.embeddings.weight.pow(2).sum(1, keepdim=True).t()
        )

        _, indices = (-dist).max(1)
        indices = indices.view(B, H, W)
        quantized = F.embedding(indices, self.embeddings.weight)

        emb_loss = self.commitment_cost * (quantized.detach() - x).pow(2).mean() + (quantized - x.detach()).pow(2).mean()
        emb_loss *= self.kld_scale

        quantized = x + (quantized - x).detach()
        quantized = quantized.permute(0, 3, 1, 2) #.reshape(B, C, H, W)

        quantized = self.out_proj(quantized)

        return quantized, emb_loss