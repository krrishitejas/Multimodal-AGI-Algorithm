import torch
import torch.nn as nn
import numpy as np

class SineLayer(nn.Module):
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

class ManifoldAcoustic(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=512):
        super().__init__()
        
        # 1. Time Mapper (1D)
        # Maps time 't' to internal features. 
        # High omega_0 because audio has very high frequency (20Hz - 20kHz)
        self.time_mapper = SineLayer(1, 128, is_first=True, omega_0=3000.0) 
        
        # 2. Latent Mapper (The "Voice/Tone" Input)
        self.latent_mapper = nn.Linear(latent_dim, 128)
        
        # 3. The Resonant Body
        self.net = nn.Sequential(
            SineLayer(256, hidden_dim),
            SineLayer(hidden_dim, hidden_dim),
            SineLayer(hidden_dim, hidden_dim),
            SineLayer(hidden_dim, hidden_dim)
        )
        
        # 4. The Output Head (Amplitude)
        # Outputs a single scalar (-1.0 to 1.0) representing air pressure
        self.amplitude_head = nn.Linear(hidden_dim, 1)

    def forward(self, time_coords, latent_code):
        """
        time_coords: (Batch, Num_Samples, 1) -> Time points (0.0 to 1.0)
        latent_code: (Batch, Latent_Dim) -> The conditioning vector (e.g., "A-sharp pitch")
        """
        B, N, _ = time_coords.shape
        
        # Process Time
        time_feat = self.time_mapper(time_coords)
        
        # Process Latents
        latent_feat = self.latent_mapper(latent_code).unsqueeze(1).expand(-1, N, -1)
        
        # Fuse
        x = torch.cat([time_feat, latent_feat], dim=-1)
        
        # Compute Waveform
        features = self.net(x)
        
        # Tanh activation to ensure output is between -1 and 1 (standard audio range)
        amplitude = torch.tanh(self.amplitude_head(features))
        
        return amplitude
