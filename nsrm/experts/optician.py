import torch
import torch.nn as nn
import numpy as np

class SineLayer(nn.Module):
    """
    Standard SIREN layer for high-frequency image details.
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class ManifoldOptician(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=512):
        super().__init__()
        
        # 1. Coordinate Mapper (2D Space)
        # Maps (x,y) pixel coordinates to internal features
        self.coord_mapper = SineLayer(2, 128, is_first=True)
        
        # 2. Latent Mapper (The "Idea")
        self.latent_mapper = nn.Linear(latent_dim, 128)
        
        # 3. The Resonant Body
        # Deep network to learn complex textures/lighting
        self.net = nn.Sequential(
            SineLayer(256, hidden_dim),
            SineLayer(hidden_dim, hidden_dim),
            SineLayer(hidden_dim, hidden_dim),
            SineLayer(hidden_dim, hidden_dim),
            SineLayer(hidden_dim, hidden_dim) # Deeper than Geometer for texture detail
        )
        
        # 4. The Output Head (RGB)
        self.rgb_head = nn.Linear(hidden_dim, 3)

    def forward(self, coords, latent_code):
        """
        coords: (Batch, Num_Pixels, 2) -> (x,y) positions normalized [-1, 1]
        latent_code: (Batch, Latent_Dim) -> The conditioning vector
        """
        B, N, _ = coords.shape
        
        # Process Coordinates
        coord_feat = self.coord_mapper(coords)
        
        # Process Latents
        latent_feat = self.latent_mapper(latent_code).unsqueeze(1).expand(-1, N, -1)
        
        # Fuse
        x = torch.cat([coord_feat, latent_feat], dim=-1)
        
        # Compute Image Function
        features = self.net(x)
        
        # Output RGB (Sigmoid ensures 0.0 - 1.0 range)
        rgb = torch.sigmoid(self.rgb_head(features))
        
        return rgb
