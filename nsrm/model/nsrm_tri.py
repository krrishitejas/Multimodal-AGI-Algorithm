import torch
import torch.nn as nn
from nsrm.graph.hypergraph import RecursiveHypergraph
from nsrm.experts.geometer import ManifoldGeometer
from nsrm.experts.optician import ManifoldOptician
from nsrm.experts.acoustic import ManifoldAcoustic
from nsrm.bridge.router import SemanticRouter

class NSRM_Tri_Mind(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Executive Core
        self.logic_core = RecursiveHypergraph(node_dim=64)
        
        # 2. Router (Now with 3 Outputs)
        # [Geometer, Optician, Acoustic]
        self.router = SemanticRouter(graph_dim=64, expert_latent_dim=16, num_experts=3)
        
        # 3. The Manifold Experts
        self.expert_geometer = ManifoldGeometer(latent_dim=16) # 3D
        self.expert_optician = ManifoldOptician(latent_dim=16) # Image
        self.expert_acoustic = ManifoldAcoustic(latent_dim=16) # Audio
        
    def forward(self, user_intent, coords_3d=None, coords_2d=None, coords_1d=None):
        # A. Reasoning
        graph_state = self.logic_core(user_intent)
        weights, thought_vector = self.router(graph_state)
        
        # Weights: [Geo, Opt, Audio]
        w_geo = weights[:, 0].view(-1, 1, 1)
        w_opt = weights[:, 1].view(-1, 1, 1)
        w_aud = weights[:, 2].view(-1, 1, 1)
        
        out = {"router_weights": weights}
        
        # B. Execution
        if coords_3d is not None:
            raw_sdf, raw_rgb = self.expert_geometer(coords_3d, thought_vector)
            out["sdf"] = raw_sdf * w_geo
            
        if coords_2d is not None:
            raw_img = self.expert_optician(coords_2d, thought_vector)
            out["image"] = raw_img * w_opt
            
        if coords_1d is not None:
            # Acoustic Expert
            raw_audio = self.expert_acoustic(coords_1d, thought_vector)
            out["audio"] = raw_audio * w_aud
            
        return out
