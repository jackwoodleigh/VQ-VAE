import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from VQVAE import VQVAE, ModelHandler

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize shorter side to 256
        transforms.CenterCrop(64),  # Then center crop to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    training_set = datasets.ImageFolder(root='imagenette/train', transform=transform)
    '''subset_indices = list(range(256))  # Indices for the first 100 examples
    training_set = Subset(training_set, subset_indices)'''

    training_loader = DataLoader(training_set, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)

    model = VQVAE(512, 256, 0.25).to("cuda")
    helper = ModelHandler(model, learning_rate=0.0001)
    model.print_parameter_count()
    helper.training(training_loader, 50, marco_batch_count=64)
