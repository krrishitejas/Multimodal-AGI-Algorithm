import torch
import torch.nn as nn
from nsrm.graph.hypergraph import RecursiveHypergraph
from nsrm.experts.geometer import ManifoldGeometer
from nsrm.experts.optician import ManifoldOptician
from nsrm.bridge.router import SemanticRouter

class NSRM_Dual_Mind(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. The Executive Core (Reasoning)
        self.logic_core = RecursiveHypergraph()
        
        # 2. The Nervous System (Router)
        # Classifies intent into 2 streams: [Geometer, Optician]
        self.router = SemanticRouter(graph_dim=64, expert_latent_dim=16, num_experts=2)
        
        # 3. The Manifold Experts
        self.expert_geometer = ManifoldGeometer(latent_dim=16) # 3D
        self.expert_optician = ManifoldOptician(latent_dim=16) # 2D
        
    def forward(self, user_intent_embedding, coords_3d=None, coords_2d=None):
        """
        Handling multimodal inputs in a single forward pass.
        """
        # Step A: Reasoning & Routing
        # graph_state = self.logic_core(user_intent_embedding) 
        # For this training, input is the embedding vector directly
        graph_state = user_intent_embedding
        
        weights, thought_vector = self.router(graph_state)
        
        # Weights: [Prob_Geometer, Prob_Optician]
        geometer_conf = weights[:, 0].view(-1, 1, 1)
        optician_conf = weights[:, 1].view(-1, 1, 1)
        
        # Step B: Execution (Parallel Processing)
        # We run both, but mask the output based on router confidence.
        # This allows soft-switching or hybrid outputs.
        
        sdf_out = None
        rgb_3d_out = None
        rgb_2d_out = None
        
        # 3D Pathway
        if coords_3d is not None:
            raw_sdf, raw_rgb_3d = self.expert_geometer(coords_3d, thought_vector)
            # Modulate by confidence: If confidence is 0, output is 0 (or masked later)
            # We return raw mostly, weighted by conf if needed
            sdf_out = raw_sdf * geometer_conf
            rgb_3d_out = raw_rgb_3d
            
        # 2D Pathway
        if coords_2d is not None:
            raw_rgb_2d = self.expert_optician(coords_2d, thought_vector)
            rgb_2d_out = raw_rgb_2d * optician_conf
            
        return {
            "sdf": sdf_out,
            "rgb_3d": rgb_3d_out,
            "image": rgb_2d_out,
            "router_weights": weights
        }
