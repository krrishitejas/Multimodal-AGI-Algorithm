import torch
import torch.nn as nn
from nsrm.csp.fno import FNOBlock
from nsrm.spectral.gfnet import GlobalFilterLayer
from nsrm.spectral.resonance import LearnableBandPassFilter

class NSRMBlock(nn.Module):
    """
    Building block of the Resonant Manifold.
    Integrates Spectral Processing (FNO/GFNet) with standard Transformer-like MLP/Norm structure.
    """
    def __init__(self, dim, modes=None, seq_len=None):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        
        # Spectral Path
        # Toggles between FNO (truncated modes) and GFNet (full global filter)
        self.use_fno = modes is not None
        
        if self.use_fno:
            self.spectral_op = FNOBlock(dim, modes)
        else:
            self.spectral_op = GlobalFilterLayer(dim, seq_len if seq_len else 512)
            
        # Optional: We could integrate BandPass here, 
        # but GFNet's learnable weights act as a general filter already.
        
        # MLP Path
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        # x: (B, L, D) - Standard format
        # Spectral Ops expect: (B, D, L)
        
        shortcut = x
        x_norm = self.norm1(x)
        x_trans = x_norm.transpose(1, 2) # (B, D, L)
        
        if self.use_fno:
            x_spec = self.spectral_op(x_trans)
        else:
            x_spec = self.spectral_op(x_trans)
        
        x_out = x_spec.transpose(1, 2) # Back to (B, L, D)
        
        x = shortcut + x_out
        
        # MLP Block
        x = x + self.mlp(self.norm2(x))
        return x
