import torch
import torch.nn as nn

class LearnableBandPassFilter(nn.Module):
    """
    Applies a soft, learnable band-pass filter in the frequency domain.
    Useful for 'tuning in' to specific semantic frequencies.
    """
    def __init__(self, channels):
        super().__init__()
        # Learnable low cut-off and bandwidth for each channel
        # Initialize to random bands
        self.low = nn.Parameter(torch.rand(channels, 1) * 0.5)
        self.band = nn.Parameter(torch.rand(channels, 1) * 0.5)
        
    def forward(self, x_fft):
        """
        x_fft: (Batch, Channels, Freq) - Complex tensor
        Returns: Filtered x_fft
        """
        freq_dim = x_fft.shape[-1]
        # Create normalized frequency grid [0, 1]
        freq_grid = torch.linspace(0, 1, freq_dim, device=x_fft.device).view(1, 1, -1)
        
        # Sharpness of the filter edges (simulating rectangular window but differentiable)
        scale = 100.0 
        
        # Ensure bandwidth is positive
        actual_band = torch.abs(self.band)
        high = self.low + actual_band
        
        # Soft mask: Sigmoid(f - low) * Sigmoid(high - f)
        # Approaches 1 inside [low, high], 0 outside
        mask = torch.sigmoid(scale * (freq_grid - self.low)) * torch.sigmoid(scale * (high - freq_grid))
        
        # Apply mask to real and imag parts
        # mask is (1, C, F) - real
        # x_fft is complex
        return x_fft * mask.type_as(x_fft.real)
