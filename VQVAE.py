import torch
from torch import nn
from Blocks import ResBlockBuilder
from torch.nn import functional as F

class Encoder(nn.Module):

    def __init__(self, input_channels, d_model, block_structure, scale_structure, block_multiplier):
        super().__init__()
        self.layers = ResBlockBuilder(-1, d_model, block_structure, scale_structure, block_multiplier)
        self.input_conv = nn.Conv2d(input_channels, d_model, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.input_conv(x)
        for module in self.layers:
            x = module(x)
        return x

class Decoder(nn.Module):

    def __init__(self, out_channels, d_model, block_structure, scale_structure, block_multiplier):
        super().__init__()
        self.layers = ResBlockBuilder(1, d_model, block_structure, scale_structure, block_multiplier)
        self.output_conv = nn.Conv2d(d_model, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        for module in self.layers:
            x = module(x)
        return self.output_conv(x)


class Quantizer(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim, beta):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embeddings_dim = embeddings_dim
        self.beta = beta

        self.codebook_embedding = nn.Embedding(num_embeddings, embeddings_dim)
        self.codebook_embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, x):
        inputs = x
        shape = x.shape
        B, C, H, W = shape

        # (B, C, H, W) -> (B, H, W, C) -> (N, emb_dim)
        x = x.view(B, H, W, C).view(-1, self.embeddings_dim)

        # find the distance between input and codebook
        # inputs = (N, 1, emb_dim), code book = (1, num_emb, emb_dim)
        # pairwise norm = (N, 1, emb_dim) -> (N, num_emb, emb_dim) -> (N, num_emb)
        x = (x.unsqueeze(1) - self.codebook_embedding.weight.unsqueeze(0)) ** 2
        distances = x.sum(dim=2)

        # Find the index of the minimum distance
        # (N, 1) closest index in code book
        indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # get codebook values from embedding layer
        quantized = self.codebook_embedding(indices).view(shape)

        # detach quantized from codebook to encourages encoder outputs to move closer to codebook
        encoder_loss = F.mse_loss(quantized.detach(), inputs)

        # detach inputs from encoder so gradient flows to codebook encouraging vectors to be close to encoder outputs
        commitment_loss = F.mse_loss(inputs.detach(), quantized)

        # loss formulation
        embedding_loss = encoder_loss + self.beta * commitment_loss

        # straight through estimator
        # set quantized to be input tensor plus the difference such that it is now the input tensor with quantized values
        # detach difference so that gradient doesn't flow to codebook
        quantized = inputs + (quantized - inputs).detach()

        # return to original shape
        quantized = quantized.view(shape)

        return quantized, embedding_loss


class VQVAE(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim, beta, device="cuda"):
        super().__init__()
        self.device = device
        self.quantizer = Quantizer(num_embeddings, embeddings_dim, beta)
        self.encoder = Encoder(input_channels=3, d_model=64, block_structure=[1, 1], scale_structure=[0, 1], block_multiplier=[1, 2])
        self.decoder = Decoder(out_channels=3, d_model=64, block_structure=[1, 1], scale_structure=[0, 1], block_multiplier=[1, 2])

    def forward(self, x):
        x = self.encoder(x)
        x, loss = self.quantizer(x)
        x = self.decoder(x)
        return x, loss


class ModelHandler:
    def __init__(self, model, lr, device="cuda"):
        self.device = device
        self.model = model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr)

    def training(self, training_loader, epoch):
        self.model.train()

        for e in range(epoch):
            for images, labels in training_loader:
                images = images.to(self.device)
                self.optimizer.zero_grad()

                pred_images, embedding_loss = self.model(images)

                reconstruction_loss = F.mse_loss(pred_images, images)

                loss = reconstruction_loss + embedding_loss
                print(loss.item())
                loss.backward()
                self.optimizer.step()


'''
vae = VQVAE(64, 64, 0.5)
t = torch.randn(1, 3, 16, 16)
print(vae(t).shape)
'''
