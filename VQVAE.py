import torch
from torch import nn
from VQVAE_components import Encoder, Decoder, Quantizer
from torch.nn import functional as F
from Blocks import ResBlock
from sklearn.cluster import KMeans
import numpy as np
from torch.utils.data import Subset
import math

class VQVAE(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim, beta, device="cuda"):
        super().__init__()
        self.device = device
        self.beta = beta
        self.quantizer = Quantizer(num_embeddings=num_embeddings, embeddings_dim=embeddings_dim)
        self.encoder = Encoder(
            d_model=64,
            in_channels=32,
            mid_channels=64,
            out_channels=1,
            res_structure=[2],
            res_multiplier=[2],
            scale_structure=[1]
        )

        self.decoder = Decoder(
            d_model=64,
            in_channels=1,
            mid_channels=64,
            out_channels=32,
            res_structure=[2],
            res_multiplier=[2],
            scale_structure=[1]
        )

        self.loss = torch.nn.MSELoss()
        self.commitment_weight = nn.Parameter(torch.zeros(1))

    def forward(self, images):
        self.train()
        x = self.encoder(images)
        emb_loss, quantized, perplexity = self.quantizer(x, self.beta)
        pred_images = self.decoder(x)

        reconstruction_loss = self.loss(pred_images, images)
        #commitment_weight = torch.clamp(self.commitment_weight, min=0.01, max=5.0)

        loss = reconstruction_loss + emb_loss

        return pred_images, loss, perplexity

    def k_means_init(self, training_loader, init_iters_count=50):
        latents = []
        i = 0
        print("Initializing...")
        with torch.no_grad():
            for image, _ in training_loader:
                print(i, end=" ")
                image = image.to(self.device)
                latents.append(self.encoder(image).to("cpu"))
                i += 1
                if i == init_iters_count:
                    break

        latents = torch.cat(latents, dim=0)
        latents = latents.view(-1, self.quantizer.embeddings_dim)
        kmeans = KMeans(n_clusters=self.quantizer.num_embeddings, n_init=10)
        kmeans.fit(latents.numpy())

        centroids = torch.from_numpy(kmeans.cluster_centers_).float().to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)

        self.quantizer.codebook.weight = centroids.T

    def print_parameter_count(self):
        print(sum(p.numel() for p in self.parameters()))


class ModelHandler:
    def __init__(self, model, learning_rate, device="cuda"):
        self.device = device
        self.model = model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.99), fused=True)
        self.loss = torch.nn.MSELoss()

    def training(self, training_loader, epoch, total_batch_size=4):
        #self.model.k_means_init(training_loader)
        current_step = 0
        acc_loss = 0
        perplexity_avg = 0

        micro_batch_size = training_loader.batch_size
        macro_batch_size = 1
        if total_batch_size is not None:
            macro_batch_size = total_batch_size // micro_batch_size

        for e in range(epoch):
            print(f"Epoch {e}")
            for images, labels in training_loader:
                images = images.to(self.device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_images, loss, perplexity = self.model(images)
                    loss = loss / macro_batch_size

                acc_loss += loss.detach()
                perplexity_avg += perplexity / macro_batch_size

                loss.backward()
                current_step += 1

                if current_step == macro_batch_size:
                    torch.cuda.synchronize()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    print(f"Accumulated loss: {acc_loss}, Perplexity: {perplexity_avg}")
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    acc_loss = 0
                    current_step = 0
                    perplexity_avg = 0

            self.save_model("model_save.pt")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print("Model Saved.")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        print("Model Loaded.")


'''vae = VQVAE(64, 64, 0.5)
t = torch.randn(1, 3, 16, 16)
t2, _ = vae(t)
print(t2.shape)'''
