import torch
from torch import nn
from Blocks import ResLayerBuilder
from torch.nn import functional as F

class Encoder(nn.Module):

    def __init__(self, input_channels, d_model, block_structure, scale_structure, block_multiplier):
        super().__init__()
        self.layers = ResLayerBuilder(-1, d_model, block_structure, scale_structure, block_multiplier)
        self.input_conv = nn.Conv2d(input_channels, d_model, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.input_conv(x)
        for module in self.layers:
            x = module(x)
        return x

class Decoder(nn.Module):

    def __init__(self, out_channels, d_model, block_structure, scale_structure, block_multiplier):
        super().__init__()
        self.layers = ResLayerBuilder(1, d_model, block_structure, scale_structure, block_multiplier)
        self.output_conv = nn.Conv2d(d_model, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        for module in self.layers:
            x = module(x)
        return self.output_conv(x)


class Quantizer(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim, decay=0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embeddings_dim = embeddings_dim
        self.decay = decay
        self.eps=1e-5

        self.register_buffer("codebook", torch.randn(embeddings_dim, num_embeddings))
        self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer("ema_input_sum_per_embedding", self.codebook.clone())

    def forward(self, inputs):

        # Flatten input
        flatten = inputs.view(-1, self.embeddings_dim)

        # Compute distance from encoder outputs to codebook vectors
        l2_input = torch.sum(flatten.pow(2), dim=1, keepdim=True)
        input_dot_codebook = flatten @ self.codebook
        l2_codebook = torch.sum(self.codebook.pow(2), dim=0, keepdim=True)

        distances = l2_input - 2*input_dot_codebook + l2_codebook

        # finding the nearest codebook vector indices and one hot it
        nearest_codebook_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        nearest_codebook_indices_one_hot = F.one_hot(nearest_codebook_indices, self.num_embeddings).squeeze(1).type(inputs.dtype)

        # updating codebook vectors with EMA
        if self.training:
            # total number of uses for each embedding
            nearest_codebook_indices_one_hot_sum = nearest_codebook_indices_one_hot.sum(dim=0)

            # sum of encoder output vectors assigned to each codebook vector
            input_sum_per_embedding = flatten.transpose(0, 1) @ nearest_codebook_indices_one_hot

            # updating ema of cluster size
            self.ema_cluster_size.data.mul_(self.decay).add_((1-self.decay) * nearest_codebook_indices_one_hot_sum)
            self.ema_input_sum_per_embedding.data.mul_(self.decay).add_((1-self.decay) * input_sum_per_embedding)

            # total size of all clusters
            n = self.ema_cluster_size.sum()

            # adds small number eps to all clusters for no division by 0
            # normalizes cluster+eps by new total sum that includes eps
            norm = (self.ema_cluster_size + self.eps) / (n + self.num_embeddings * self.eps)
            # denormalizes but now there are no 0's so division works
            cluster_size = norm * n

            # moving codebook clusters by mean
            input_mean_per_embedding = self.ema_input_sum_per_embedding / cluster_size.unsqueeze(0)
            self.codebook.data.copy_(input_mean_per_embedding)

        # use the one hot encodings for nearest vectors to get codebook vectors
        quantized = (nearest_codebook_indices_one_hot @ self.codebook.t()).view(inputs.shape)

        embedding_loss = torch.mean((quantized.detach() - inputs).pow(2))

        # straight through estimator
        quantized = inputs + (quantized - inputs).detach()

        return quantized, embedding_loss


class VQVAE(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim, beta, device="cuda"):
        super().__init__()
        self.device = device
        self.beta = beta
        self.quantizer = Quantizer(num_embeddings, embeddings_dim)
        self.encoder = Encoder(input_channels=3, d_model=64, block_structure=[4], scale_structure=[1], block_multiplier=[2])
        self.decoder = Decoder(out_channels=3, d_model=64, block_structure=[4], scale_structure=[1], block_multiplier=[2])

        self.loss = torch.nn.MSELoss()

    def forward(self, images):
        self.train()

        x = self.encoder(images)
        x, embedding_loss = self.quantizer(x)
        pred_images = self.decoder(x)

        reconstruction_loss = self.loss(pred_images, images)
        loss = reconstruction_loss + self.beta * embedding_loss

        return pred_images, loss

    def decode_image(self, x):
        self.eval()
        with torch.no_grad():
            x, _ = self.quantizer(x)
            x = self.decoder(x)
        return x

    def encode_image(self, x):
        self.eval()
        with torch.no_grad():
            x = self.encoder(x)
        return x

    def print_parameter_count(self):
        print(sum(p.numel() for p in self.parameters()))


class ModelHandler:
    def __init__(self, model, learning_rate, device="cuda"):
        self.device = device
        self.model = model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, fused=True)

    def training(self, training_loader, epoch, marco_batch_count=4):

        current_step = 0
        acc_loss = 0
        B = training_loader.batch_size
        norm_val = 1.
        if marco_batch_count is not None:
            norm_val = marco_batch_count // B

        for e in range(epoch):
            print(f"Epoch {e}")
            for images, labels in training_loader:
                images = images.to(self.device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_images, loss = self.model(images)

                loss = loss / norm_val
                acc_loss += loss.detach()
                loss.backward()
                current_step += 1

                if current_step == marco_batch_count:
                    torch.cuda.synchronize()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    print(acc_loss/B)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    acc_loss = 0
                    current_step = 0



'''vae = VQVAE(64, 64, 0.5)
t = torch.randn(1, 3, 16, 16)
t2, _ = vae(t)
print(t2.shape)'''
