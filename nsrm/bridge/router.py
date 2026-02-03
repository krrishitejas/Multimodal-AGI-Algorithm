import torch
import torch.nn as nn
import numpy as np

class SemanticRouter(nn.Module):
    """
    The Expert Router.
    Decides WHO does the job (Routing) and WHAT they should do (Transduction).
    """
    def __init__(self, graph_dim, expert_latent_dim, num_experts=3):
        super().__init__()
        
        # 1. The Gating Mechanism (Who?)
        # Analyzes the graph state to decide if this is a 3D, Video, or Text task.
        self.gate = nn.Sequential(
            nn.Linear(graph_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts), # Output: Probability for [Geometer, Optician, Logician]
            nn.Softmax(dim=-1)
        )
        
        # 2. The Transducer (What?)
        # Converts the discrete graph embedding into a continuous "Thought Vector"
        # for the expert.
        self.transducer = nn.Sequential(
            nn.Linear(graph_dim, 128),
            SineActivation(), # Semantic Resonance (keeping the signal continuous)
            nn.Linear(128, expert_latent_dim)
        )

    def forward(self, graph_embedding):
        """
        Input: graph_embedding (Batch, Graph_Dim) -> The summary state of the reasoning graph.
        """
        # A. Who? (Routing Weights)
        router_logits = self.gate(graph_embedding)
        
        # B. What? (Conditioning Vector)
        latent_instruction = self.transducer(graph_embedding)
        
        return router_logits, latent_instruction

class SineActivation(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.sin(x)
