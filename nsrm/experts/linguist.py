import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Ensure we can import from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from nsrm.csp.fno import SpectralConv1d

class StateSpaceBlock(nn.Module):
    """
    A simplified State Space Model (SSM) block (Mamba-style).
    Uses a Selective Scan mechanism approximated via Gated Linear Recurrence.
    """
    def __init__(self, dim, stats_expand=2):
        super().__init__()
        hidden_dim = dim * stats_expand
        
        # Projections
        self.in_proj = nn.Linear(dim, hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim, dim)
        
        # SSM Parameters (Simplified A, B, C, D)
        # A_log: Decay rates
        self.A_log = nn.Parameter(torch.log(torch.arange(1, hidden_dim + 1, dtype=torch.float32)).unsqueeze(1))
        self.D = nn.Parameter(torch.ones(hidden_dim))
        
        # 1D Conv for local context (like Mamba)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        
        self.act = nn.SiLU()

    def forward(self, x, state=None):
        # x: (Batch, Seq_Len, Dim)
        B, L, D = x.shape
        
        # 1. Input Projection
        xz = self.in_proj(x) # (B, L, 2*H)
        x_proj, z = xz.chunk(2, dim=-1) # x_proj: Signal, z: Gate
        
        # 2. Convolution (Local Context)
        # Permute for Conv1d: (B, H, L)
        x_conv = self.conv(x_proj.transpose(1, 2)).transpose(1, 2)
        x_conv = self.act(x_conv)
        
        # 3. SSM (Approximated via Element-wise recurrence / Gating)
        # For this prototype, we use a simple Gated Linear Unit with decay (A) simulation
        # In a real Mamba, this is a parallel scan. Here we use a residual gating for speed/simplicity
        # or just the Conv result + a long-range skip. 
        # To respect the "trajectory" concept, we'll assume the FNO handled long-range global reasoning.
        # This block handles local state transitions.
        
        y = x_conv * F.silu(z) # Simple Gated output ("Mamba-lite")
        
        # 4. Output Projection
        out = self.out_proj(y)
        
        return out

class ManifoldLinguist(nn.Module):
    def __init__(self, latent_dim=256, vocab_size=50000, hidden_dim=512):
        super().__init__()
        
        # 1. The "Thought" Processor (Spectral FNO)
        # It processes the reasoning vector as a waveform
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.thought_processor = nn.Sequential(
            nn.SiLU(),
            # FNO Layer (Frequency Domain Processing)
            # modes=32 implies capturing low-frequency semantic trends
            SpectralConv1d(hidden_dim, hidden_dim, modes=32) 
        )
        
        # 2. The Trajectory Generator (SSM / Mamba-style)
        # Generates the flow of the sentence
        self.trajectory_gen = StateSpaceBlock(dim=hidden_dim)
        
        # 3. The "De-Projector" (Meaning -> Words)
        # Maps the continuous meaning back to discrete vocabulary
        self.vocab_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, thought_vector, seq_len=16):
        """
        thought_vector: (Batch, Latent_Dim)
        seq_len: Length of text sequence to generate
        """
        B = thought_vector.shape[0]
        
        # Expand thought into a sequence/trajectory placeholder
        # We repeat the thought across time, but allow FNO to warp it into a trajectory
        x = self.input_proj(thought_vector).unsqueeze(2) # (B, H, 1)
        x = x.repeat(1, 1, seq_len) # (B, H, L)
        
        # Step 1: Resonate on the Thought (Global Frequency Mixing)
        # "What is the logical conclusion of this thought?"
        # Input to FNO must be (B, C, L)
        deep_thought = self.thought_processor(x) # (B, H, L)
        
        # Transpose for SSM (B, L, H)
        deep_thought = deep_thought.transpose(1, 2)
        
        # Step 2: Generate Trajectory (Local Dynamics)
        # "How does this thought unfold in time?"
        semantic_flow = self.trajectory_gen(deep_thought)
        
        # Step 3: Project to Language
        word_logits = self.vocab_head(semantic_flow) # (B, L, Vocab)
        
        return word_logits
