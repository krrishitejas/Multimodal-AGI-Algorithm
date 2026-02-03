import torch
import torch.nn as nn
import numpy as np

class FourierFeatureMapping(nn.Module):
    """
    Projects input coordinates into high-dimensional Fourier feature space.
    Based on 'Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains' (Tancik et al., 2020).
    """
    def __init__(self, input_dim, mapping_size, scale=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.scale = scale
        # B matrix for projection: (mapping_size, input_dim)
        # Sampled from Gaussian N(0, scale^2)
        # We assume coordinates are normalized (e.g., [0, 1] or [-1, 1])
        self.register_buffer('B', torch.randn((mapping_size, input_dim)) * scale)

    def forward(self, x):
        """
        x: (..., input_dim) input coordinates
        Returns: (..., 2 * mapping_size) sine and cosine features
        """
        # x projected: (..., mapping_size)
        # 2 * pi * B * x
        projected = (2 * np.pi * x) @ self.B.t()
        
        # [sin(proj), cos(proj)]
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)

class INRNetwork(nn.Module):
    """
    Implicit Neural Representation Network.
    Maps coordinates to semantic values/signals via an MLP with Fourier features.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, fourier_mapping_size=256, fourier_scale=10.0):
        super().__init__()
        self.fourier_mapping = FourierFeatureMapping(input_dim, fourier_mapping_size, fourier_scale)
        
        layers = []
        # Input dimension after fourier mapping is 2 * mapping_size
        in_dim = 2 * fourier_mapping_size
        
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU()) # GELU is often better for continuous functions
            in_dim = hidden_dim
            
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        # x is coordinate (e.g., time t, or pos x,y)
        # returns value at that coordinate
        x_proj = self.fourier_mapping(x)
        return self.net(x_proj)
