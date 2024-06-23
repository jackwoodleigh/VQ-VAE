import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from VQVAE import VQVAE, ModelHandler

transform = transforms.Compose([
    transforms.Resize(256),  # Resize shorter side to 256
    transforms.CenterCrop(224),  # Then center crop to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

training_set = datasets.ImageFolder(root='imagenette/train', transform=transform)

training_loader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=4)

model = VQVAE(512, 512, 0.5)
helper = ModelHandler(model, 0.001)