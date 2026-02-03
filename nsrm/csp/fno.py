import torch
import torch.nn as nn
import torch.fft

class SpectralConv1d(nn.Module):
    """
    1D Fourier Neural Operator layer.
    Performs global convolution in the frequency domain.
    """
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        """
        in_channels: input channels
        out_channels: output channels
        modes: number of fourier modes to multiply (low frequency truncation).
               If modes >= seq_len // 2, it acts as a global filter without truncation.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Scale for initialization
        self.scale = (1 / (in_channels * out_channels))
        # Complex weights R
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat))

    def complex_mul1d(self, input, weights):
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        # x: (Batch, Channels, Length)
        batchsize = x.shape[0]
        
        # Compute Fourier coefficients
        # rfft returns (B, C, L//2 + 1) complex numbers
        x_ft = torch.fft.rfft(x)

        # Output holder
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        
        # Multiply relevant Fourier modes
        actual_modes = min(self.modes, x_ft.size(-1))
        out_ft[:, :, :actual_modes] = self.complex_mul1d(x_ft[:, :, :actual_modes], self.weights[:, :, :actual_modes])

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNOBlock(nn.Module):
    """
    Standard FNO Block: SpectralConv + Skip Connection (W) + Activation
    """
    def __init__(self, width, modes):
        super().__init__()
        self.conv = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, 1) # pointwise linear (kernel size 1)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, width, L)
        x1 = self.conv(x)
        x2 = self.w(x)
        return self.act(x1 + x2)
