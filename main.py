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

    training_loader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    model = VQVAE(512, 64, 0.1).to("cuda")
    helper = ModelHandler(model, learning_rate=0.0003)
    helper.load_model("model_save.pt")
    model.print_parameter_count()

    helper.training(training_loader, 500, total_batch_size=64)

    '''
    
    
    
    tensor1 = next(iter(training_loader))[0][10]
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    tensor = tensor1 * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    transform = transforms.ToPILImage()
    image = transform(tensor)

    image.save('input_image.png')

    tensor = tensor1.unsqueeze(0)
    x, _, _ = model(tensor.to("cuda"))
    x = x.to("cpu").squeeze(0)

    x = x * std + mean
    x = torch.clamp(x, 0, 1)
    x = transform(x)
    x.save('output_image.png')
    

    
    '''