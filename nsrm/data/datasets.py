import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os

class RealWorldImages(Dataset):
    def __init__(self):
        # We use CIFAR-10 or ImageNet as a starter "Reality"
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1] for Siren
        ])
        
        # Ensure data directory exists
        data_dir = './data'
        os.makedirs(data_dir, exist_ok=True)
        
        self.data = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=self.transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        # NSRM Optician expects (Pixel_Coords, RGB_Values)
        # We must convert the image grid to a coordinate list on the fly
        
        _, H, W = img.shape
        # Create grid of (x, y) coordinates
        # Using indexing='ij' to match standard image coordinate systems
        y_grid, x_grid = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
        coords = torch.stack([x_grid, y_grid], dim=-1).reshape(-1, 2) # (4096, 2)
        
        # Flatten image pixels to match coordinates
        # Permute from (C, H, W) to (H, W, C) then flatten to (N_pixels, C)
        rgb = img.permute(1, 2, 0).reshape(-1, 3) # (4096, 3)
        
        return coords, rgb, label
