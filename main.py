import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from VQVAE import VQVAE, ModelHandler

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize(144),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    training_set = datasets.ImageFolder(root='imagenette/train', transform=transform)
    '''subset_indices = list(range(256))  # Indices for the first 100 examples
    training_set = Subset(training_set, subset_indices)'''

    training_loader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    model = VQVAE(512, 128, 0.001).to("cuda")
    helper = ModelHandler(model, learning_rate=0.00001)
    model.print_parameter_count()
    helper.training(training_loader, 50, total_batch_size=256)
