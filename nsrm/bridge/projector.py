import torch
import torch.nn as nn
import numpy as np

class SineLayer(nn.Module):
    """
    Linear layer with Sine activation.
    Essential for Siren Networks (Implicit Neural Representations).
    See: 'Implicit Neural Representations with Periodic Activation Functions' (Sitzmann et al., 2020)
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 
                                            1 / self.linear.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.linear.in_features) / self.omega_0, 
                                            np.sqrt(6 / self.linear.in_features) / self.omega_0)
            
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class TextToSignal(nn.Module):
    """
    Continuous Projector (Text2Manifold).
    Maps Word Position -> Signal Trajectory using a Siren Network.
    A discrete sentence is treated as a continuous curve in semantic space.
    """
    def __init__(self, embedding_dim, hidden_dim=256, num_layers=3):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Core Siren Network
        # Input: 1D coordinate (position in sentence, normalized 0-1 or similar)
        # Output: embedding_dim (semantic vector at that position)
        layers = []
        layers.append(SineLayer(1, hidden_dim, is_first=True))
        for _ in range(num_layers - 2):
            layers.append(SineLayer(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, embedding_dim)) # Final layer is linear
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, positions):
        """
        positions: (Batch, Seq_Len) or (Batch, Seq_Len, 1) - Normalized positions [-1, 1] or [0, 1]
        Returns: (Batch, Seq_Len, Embedding_Dim)
        """
        if positions.dim() == 2:
            positions = positions.unsqueeze(-1) # Ensure (B, L, 1)
            
        return self.net(positions)
