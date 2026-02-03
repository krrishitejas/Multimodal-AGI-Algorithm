# Neuro-Symbolic Resonant Manifold (NSRM) - Proof of Concept

This repository contains a technical Proof-of-Concept (PoC) for the **Neuro-Symbolic Resonant Manifold (NSRM)** architecture, illustrating a path toward AGI beyond the Transformer paradigm.

## Architecture Pillars

This implementation focuses on the three core pillars of the NSRM:

1.  **Continuous Signal Processing (CSP)**:
    - **Implicit Neural Representations (INRs)**: `nsrm.csp.inr`. Models data as continuous functions using Fourier Features.
    - **Fourier Neural Operators (FNOs)**: `nsrm.csp.fno`. Learns resolution-invariant operators in the frequency domain.

2.  **Spectral Resonance**:
    - **Global Filter Networks (GFNet)**: `nsrm.spectral.gfnet`. Uses FFT-based global convolution for log-linear mixing.
    - **Band-Pass Filters**: `nsrm.spectral.resonance`. Learnable filters to isolate semantic frequencies.

3.  **Recursive Graph Theory**:
    - **Recursive Hypergraphs**: `nsrm.graph.hypergraph`. Topological structures where edges can be nodes.
    - **Joint Rewiring**: `nsrm.graph.rewiring`. Dynamic graph evolution based on spectral correlation ("Resonance").

## Quick Start

### Prerequisites
- Python 3.9+
- PyTorch 2.0+
- NumPy, SciPy

### Installation
```bash
pip install -r requirements.txt
```

### Running Tests
To verify the architecture components:
```bash
python test_nsrm.py
```

### Usage Example
```python
import torch
from nsrm.model.nsrm_agl import NSRMModel

# Initialize the AGI core
# input_dim=10, model_dim=64, output_dim=5
model = NSRMModel(input_dim=10, model_dim=64, output_dim=5, seq_len=128)

# Forward pass (Neural Stream)
x = torch.randn(1, 128, 10) # (Batch, Seq, Dim)
output = model(x) # (Batch, 128, 5)

# Symbolic Reasoning Step
# (Assuming node embeddings are extracted/provided)
model.symbolic_update({0: torch.randn(64), 1: torch.randn(64)})
```

## Structure
- `nsrm/core`: Utilities.
- `nsrm/csp`: INRs and FNOs.
- `nsrm/spectral`: GFNets and Resonance.
- `nsrm/graph`: Hypergraphs and Rewiring.
- `nsrm/model`: Assembled NSRM Block and Model.
