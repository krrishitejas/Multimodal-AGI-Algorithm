import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

class GlobalFilterLayer(nn.Module):
    """
    Global Filter Layer (GFNet).
    Applies a learnable global filter in the frequency domain.
    Supports resolution scaling via interpolation of the filter.
    """
    def __init__(self, dim, sequence_length=512):
        super().__init__()
        self.dim = dim
        self.base_seq_len = sequence_length
        # Frequency dim for rfft is L//2 + 1
        self.base_freq_dim = sequence_length // 2 + 1
        
        # Learnable complex weights
        self.scale = 0.02
        # Store as (dim, freq_dim, 2) for real/imag parts to ensure parameter registration works easily
        self.weights = nn.Parameter(self.scale * torch.randn(dim, self.base_freq_dim, 2))

    def forward(self, x):
        # x: (Batch, Channels, Length)
        B, C, L = x.shape
        
        # FFT
        x_fft = torch.fft.rfft(x, n=L)
        current_freq_dim = x_fft.shape[-1]
        
        # Construct complex weight (C, Freq_Base)
        w = torch.view_as_complex(self.weights)
        
        # Interpolate filter if sequence length changed
        if current_freq_dim != self.base_freq_dim:
            # Reshape for interpolation: (1, C, Freq)
            w_real = F.interpolate(w.real.unsqueeze(0), size=current_freq_dim, mode='linear', align_corners=True)
            w_imag = F.interpolate(w.imag.unsqueeze(0), size=current_freq_dim, mode='linear', align_corners=True)
            w = torch.complex(w_real.squeeze(0), w_imag.squeeze(0)) # (C, Current_Freq)
        
        # Apply Filter
        # x_fft: (B, C, F)
        # w: (C, F)
        out_fft = x_fft * w.unsqueeze(0)
        
        # IFFT
        x = torch.fft.irfft(out_fft, n=L)
        return x
