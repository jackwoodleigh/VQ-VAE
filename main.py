import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from VQVAE import VQVAE, ModelHandler
import matplotlib.pyplot as plt

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

    training_loader = DataLoader(training_set, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    model = VQVAE(1024, 32, 0.25).to("cuda")
    helper = ModelHandler(model, lr=0.0001, min_lr=0.00005)
    helper.load_model("model_save.pt")
    model.print_parameter_count()

    helper.training(training_loader, 200, 200, total_batch_size=64)

    '''
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Input vs Output Comparisons')

    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    to_pil = transforms.ToPILImage()

    for i in range(5):
        tensor = next(iter(training_loader))[0][i]

        input_tensor = tensor * std + mean
        input_tensor = torch.clamp(input_tensor, 0, 1)
        input_image = to_pil(input_tensor)

        x, _, _, _ = model(tensor.unsqueeze(0).to("cuda"), False)
        x = x.to("cpu").squeeze(0)
        x = x * std + mean
        x = torch.clamp(x, 0, 1)
        output_image = to_pil(x)

        axs[0, i].imshow(input_image)
        axs[0, i].set_title(f'Input {i + 1}')
        axs[0, i].axis('off')

        axs[1, i].imshow(output_image)
        axs[1, i].set_title(f'Output {i + 1}')
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('comparison.png')

    
    '''