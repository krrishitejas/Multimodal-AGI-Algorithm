import torch
import torch.nn as nn
from nsrm.graph.hypergraph import RecursiveHypergraph
from nsrm.experts.geometer import ManifoldGeometer
from nsrm.experts.optician import ManifoldOptician
from nsrm.experts.linguist import ManifoldLinguist
from nsrm.bridge.router import SemanticRouter

class NSRM_Trinity_Mind(nn.Module):
    def __init__(self, vocab_size=10000):
        super().__init__()
        
        # 1. The Executive Core (Reasoning)
        self.logic_core = RecursiveHypergraph()
        
        # 2. The Nervous System (Router)
        # Classifies intent into 3 streams: [Geometer, Optician, Linguist]
        self.router = SemanticRouter(graph_dim=64, expert_latent_dim=16, num_experts=3)
        
        # 3. The Manifold Experts
        self.expert_geometer = ManifoldGeometer(latent_dim=16) # 3D
        self.expert_optician = ManifoldOptician(latent_dim=16) # 2D
        self.expert_linguist = ManifoldLinguist(latent_dim=16, vocab_size=vocab_size) # Text
        
    def forward(self, user_intent_embedding, coords_3d=None, coords_2d=None, text_seq_len=None):
        """
        Handling multimodal inputs in a single forward pass.
        """
        # Step A: Reasoning & Routing
        graph_state = user_intent_embedding
        
        weights, thought_vector = self.router(graph_state)
        
        # Weights: [Prob_Geometer, Prob_Optician, Prob_Linguist]
        geometer_conf = weights[:, 0].view(-1, 1, 1)
        optician_conf = weights[:, 1].view(-1, 1, 1)
        linguist_conf = weights[:, 2].view(-1, 1, 1)
        
        # Step B: Execution (Triple Pathway)
        
        sdf_out = None
        rgb_3d_out = None
        rgb_2d_out = None
        text_logits_out = None
        
        # 3D Pathway
        if coords_3d is not None:
            raw_sdf, raw_rgb_3d = self.expert_geometer(coords_3d, thought_vector)
            sdf_out = raw_sdf * geometer_conf
            rgb_3d_out = raw_rgb_3d
            
        # 2D Pathway
        if coords_2d is not None:
            raw_rgb_2d = self.expert_optician(coords_2d, thought_vector)
            rgb_2d_out = raw_rgb_2d * optician_conf
            
        # Text Pathway (Linguist)
        # Always run if weights suggest it, or forced by input loop
        # Here we run if text_seq_len is provided
        if text_seq_len is not None:
            raw_logits = self.expert_linguist(thought_vector, seq_len=text_seq_len)
            text_logits_out = raw_logits # We don't multiply logits by confidence directly usually, 
                                         # but we could gate them. For now, raw logits.
            
        return {
            "sdf": sdf_out,
            "rgb_3d": rgb_3d_out,
            "image": rgb_2d_out,
            "text_logits": text_logits_out,
            "router_weights": weights
        }
