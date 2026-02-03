import torch
import torch.nn as nn

class ManifoldVisionEncoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        
        # 1. The Retina (Feature Extractor)
        # Compresses 64x64 images down to high-level features
        self.features = nn.Sequential(
            # Input: (3, 64, 64)
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), # -> (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> (128, 8, 8)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # -> (256, 4, 4)
            nn.ReLU(),
        )
        
        # 2. The Optic Nerve (Projection to Latent Space)
        # Flattens and maps to the 16-dim vector the Brain understands
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim) 
            # Output is now compatible with the Geometer/Linguist!
        )

    def forward(self, image_tensor):
        """
        Input: (Batch, 3, 64, 64) -> Normalized image [-1, 1]
        Output: (Batch, Latent_Dim) -> The "Concept Vector"
        """
        x = self.features(image_tensor)
        latent_code = self.projection(x)
        return latent_code
