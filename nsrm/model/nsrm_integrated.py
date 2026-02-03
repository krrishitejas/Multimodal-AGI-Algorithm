import torch
import torch.nn as nn
from nsrm.graph.hypergraph import RecursiveHypergraph
from nsrm.experts.geometer import ManifoldGeometer
from nsrm.bridge.router import SemanticRouter

class NSRM_Integrated(nn.Module):
    """
    The Integrated Neuro-Symbolic Resonant Manifold.
    Combines Logic Core (Brain), Router (Nervous System), and Geometer (Body).
    """
    def __init__(self):
        super().__init__()
        
        # 1. The Brain (Reasoning)
        # Note: In this Conditional Reality experiment, we simulate the "output" of the graph
        # by passing in concept vectors directly. 
        # But we keep the structure ready for full integration.
        self.logic_core = RecursiveHypergraph() # Not used directly in forward pass of this specific exp
        
        # 2. The Nervous System (Router)
        # Input dim 64 (Graph/Concept dim), Expert Latent dim 16
        self.router = SemanticRouter(graph_dim=64, expert_latent_dim=16)
        
        # 3. The Body (Experts)
        # We currently have one active expert: The Geometer
        self.expert_geometer = ManifoldGeometer(latent_dim=16)
        
    def forward(self, user_intent_embedding, spatial_coords):
        """
        user_intent: "Make a Cube" (Embedded vector (B, 64))
        spatial_coords: XYZ points to query (B, N, 3)
        Returns: SDF, RGB, RouterWeights
        """
        # Step 1: Think (Graph Processing)
        # In this experiment, user_intent_embedding IS the graph state
        graph_state = user_intent_embedding
        
        # Step 2: Route & Transduce
        # The router reads the graph and creates instructions
        weights, thought_vector = self.router(graph_state)
        
        # Check if the Geometer is the chosen expert (Index 0)
        geometer_weight = weights[:, 0]
        
        # Step 3: Act (Expert Execution)
        # The expert generates the SDF field based on the thought vector
        # We modulate the output by the router's confidence (weight)
        sdf, rgb = self.expert_geometer(spatial_coords, thought_vector)
        
        # We multiply SDF by weight? 
        # If weight is 0, SDF is 0? That implies "surface everywhere" (SDF=0).
        # Better: If not chosen, maybe output is ignored. 
        # But for soft routing, we can scale.
        # However, SDF=0 is a surface. SDF=Large is empty space.
        # Let's return raw SDF and let the loss/controller handle it.
        # But the prompt says: "sdf * geometer_weight.unsqueeze(-1)"
        # Let's stick to the prompt's design, assuming the system learns to handle the scaling.
        
        return sdf * geometer_weight.unsqueeze(-1).unsqueeze(-1), rgb, weights
