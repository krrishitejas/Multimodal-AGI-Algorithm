import torch
import torch.nn as nn
import numpy as np

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features # Store this
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

class ManifoldGeometer(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=512):
        super().__init__()
        
        # 1. Coordinate Mapper (The "Space" Input)
        # Maps (x,y,z) to internal features
        # LOWER OMEGA_0: Reduces high-frequency "bumps" on flat surfaces
        self.coord_mapper = SineLayer(3, 128, is_first=True, omega_0=10.0)
        
        # 2. Latent Mapper (The "Idea" Input)
        # Maps the brain's thought vector to the expert's space
        self.latent_mapper = nn.Linear(latent_dim, 128)
        
        # 3. The Resonant Body (Deep Siren Network)
        self.net = nn.Sequential(
            SineLayer(256, hidden_dim, omega_0=15.0),
            SineLayer(hidden_dim, hidden_dim, omega_0=15.0),
            SineLayer(hidden_dim, hidden_dim, omega_0=15.0),
            SineLayer(hidden_dim, hidden_dim, omega_0=15.0)
        )
        
        # 4. The Output Heads
        # SDF Head: Outputs distance to surface (scalar)
        self.sdf_head = nn.Linear(hidden_dim, 1)
        # RGB Head: Outputs surface color (3 channels)
        self.rgb_head = nn.Linear(hidden_dim, 3)

    def forward(self, coords, latent_code):
        """
        coords: (Batch, Num_Points, 3) -> The points to query in 3D space
        latent_code: (Batch, Latent_Dim) -> The conditioning vector
        """
        B, N, _ = coords.shape
        
        # Process Coordinates
        coord_feat = self.coord_mapper(coords) # (B, N, 128)
        
        # Process Latents and expand to match number of points
        # Latent code: (B, Latent_Dim)
        # Latent feat: (B, 128) -> (B, 1, 128) -> (B, N, 128)
        latent_feat = self.latent_mapper(latent_code).unsqueeze(1).expand(-1, N, -1)
        
        # Concatenate Space + Idea
        x = torch.cat([coord_feat, latent_feat], dim=-1) # (B, N, 256)
        
        # Compute Field
        features = self.net(x)
        
        sdf = self.sdf_head(features)
        rgb = torch.sigmoid(self.rgb_head(features))
        
        return sdf, rgb
