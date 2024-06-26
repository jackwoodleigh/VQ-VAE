import torch
from torch import nn
from VQVAE_components import Block, Quantizer
from Blocks import ResBlock

class VQVAE(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim, beta, device="cuda"):
        super().__init__()
        self.device = device
        self.beta = beta
        self.quantizer = Quantizer(num_embeddings, embeddings_dim)

        self.encoder = Block(block_type=-1, in_channels=[3, 128], out_channels=[128, 32, 1], d_model=64, block_structure=[2, 2], scale_structure=[1, 1], block_multiplier=[2, 2])
        self.decoder = Block(block_type=1, in_channels=[1, 32, 128], out_channels=[128, 3], d_model=64, block_structure=[2, 2], scale_structure=[1, 1], block_multiplier=[2, 2])

        self.loss = torch.nn.MSELoss()

    def forward(self, images):
        self.train()
        x = self.encoder(images)
        x, commitment_loss = self.quantizer(x)
        pred_images = self.decoder(x)

        reconstruction_loss = self.loss(pred_images, images)
        loss = reconstruction_loss + self.beta * commitment_loss

        return pred_images, loss

    def print_parameter_count(self):
        print(sum(p.numel() for p in self.parameters()))


class ModelHandler:
    def __init__(self, model, learning_rate, device="cuda"):
        self.device = device
        self.model = model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.95), fused=True)

    def training(self, training_loader, epoch, total_batch_size=4):

        current_step = 0
        acc_loss = 0
        micro_batch_size = training_loader.batch_size
        macro_batch_size = 1

        if total_batch_size is not None:
            macro_batch_size = total_batch_size // micro_batch_size

        for e in range(epoch):
            print(f"Epoch {e}")
            for images, labels in training_loader:
                images = images.to(self.device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_images, loss = self.model(images)

                loss = loss / macro_batch_size
                acc_loss += loss.detach()
                loss.backward()
                current_step += 1

                if current_step == macro_batch_size:
                    torch.cuda.synchronize()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    print(f"Accumulated loss: {acc_loss}")
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    acc_loss = 0
                    current_step = 0



'''vae = VQVAE(64, 64, 0.5)
t = torch.randn(1, 3, 16, 16)
t2, _ = vae(t)
print(t2.shape)'''
