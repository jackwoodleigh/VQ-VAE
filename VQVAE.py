import torch
from torch import nn
from VQVAE_components import Encoder, Decoder, Quantizer
from torch.nn import functional as F
from Blocks import ResBlock
from scipy.cluster.vq import kmeans2
import numpy as np
from torch.utils.data import Subset
import math
from tqdm import tqdm
from quant import VectorQuantizer

class VQVAE(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim, beta, device="cuda"):
        super().__init__()
        self.device = device
        self.beta = beta
        #self.quantizer = Quantizer(num_embeddings=num_embeddings, embeddings_dim=embeddings_dim)
        self.quantizer = VectorQuantizer(embeddings_dim, num_embeddings, beta)
        self.encoder = Encoder(
            d_model=16,
            in_channels=32,
            mid_channels=16,
            out_channels=1,
            res_structure=[1, 5, 5, 5, 5, 5],
            res_multiplier=[8, 8, 8, 4, 2, 1],
            scale_structure=[0, 1, 1, 0, 0, 0]
        )

        self.decoder = Decoder(
            d_model=16,
            in_channels=1,
            mid_channels=16,
            out_channels=32,
            res_structure=[1, 5, 5, 5, 5, 5],
            res_multiplier=[8, 8, 8, 4, 2, 1],
            scale_structure=[0, 1, 1, 0, 0, 0]
        )

        self.loss = torch.nn.MSELoss()
        self.commitment_weight = nn.Parameter(torch.zeros(1))

    def forward(self, images, replace_dead_codes):
        self.train()
        x = self.encoder(images)
        x, emb_loss = self.quantizer(x)
        pred_images = self.decoder(x)

        reconstruction_loss = self.loss(pred_images, images)
        #commitment_weight = torch.clamp(self.commitment_weight, min=0.01, max=5.0)

        loss = reconstruction_loss + emb_loss

        return pred_images, loss, 0, reconstruction_loss

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
        centroids, labels = kmeans2(latents.numpy(), self.quantizer.num_embeddings, minit='points')

        centroids = torch.tensor(centroids, dtype=self.quantizer.codebook.weight.dtype, device=self.quantizer.codebook.weight.device)
        centroids = F.normalize(centroids, p=2, dim=1)

        self.quantizer.codebook.weight = nn.Parameter(centroids)
        self.quantizer.cluster_sizes = torch.ones_like(self.quantizer.cluster_sizes)*self.quantizer.dead_code_threshold
        print()

    def print_parameter_count(self):
        print(sum(p.numel() for p in self.parameters()))

    def kmeans22(self, training_loader, init_iters_count=50):
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
        flatten = latents.view(-1, self.quantizer.embeddings_dim)

        print('running kmeans!!')  # data driven initialization for the embeddings
        rp = torch.randperm(flatten.size(0))
        kd = kmeans2(flatten[rp[:20000]].data.cpu().numpy(), self.quantizer.num_embeddings, minit='points')
        self.quantizer.codebook.weight.data.copy_(torch.from_numpy(kd[0]))



class ModelHandler:
    def __init__(self, model, lr, min_lr, device="cuda"):
        self.device = device
        self.model = model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.99), fused=False)  # doesnt work with fused
        self.loss = torch.nn.MSELoss()
        self.lr = lr
        self.min_lr = min_lr

    def training(self, training_loader, epoch, sch_epoch, save_path="model_save.pt", total_batch_size=4):
        #self.model.kmeans22(training_loader)
        current_step = 0



        micro_batch_size = training_loader.batch_size
        macro_batch_size = 1
        if total_batch_size is not None:
            macro_batch_size = total_batch_size // micro_batch_size

        '''scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            base_lr=self.min_lr,
            max_lr=self.lr,
            mode="exp_range",
            gamma=0.995,
            step_size_up=micro_batch_size//2,
            step_size_down=micro_batch_size//2
        )'''

        for e in range(epoch):
            print(f"Epoch {e}")
            self.optimizer.zero_grad()
            total_loss = 0
            rec_loss = 0
            perplexity_avg = 0
            for i, (images, labels) in tqdm(enumerate(training_loader)):
                images = images.to(self.device)

                replace_dead_codes = False
                if (e==0 and i==0) or i == len(training_loader)-1:
                    replace_dead_codes = False

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_images, loss, perplexity, rec = self.model(images, replace_dead_codes)

                total_loss += loss.detach() / macro_batch_size
                rec_loss += rec.detach() / macro_batch_size
                perplexity_avg += perplexity / macro_batch_size

                loss.backward()
                current_step += 1

                if current_step == macro_batch_size:
                    torch.cuda.synchronize()
                    #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    #scheduler.step()
                    current_step = 0


            print(f"Total loss: {total_loss / (len(training_loader) / macro_batch_size)}, Reconstruct Loss: {rec_loss / (len(training_loader) / macro_batch_size)} Perplexity: {perplexity_avg / (len(training_loader) / macro_batch_size)}") #, Learning rate: {scheduler.get_last_lr()[0]}
            acc_loss = 0
            perplexity_avg = 0
            rec_loss = 0
            self.save_model(save_path)

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
