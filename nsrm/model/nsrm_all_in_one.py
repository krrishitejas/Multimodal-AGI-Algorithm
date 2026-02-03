import torch
import torch.nn as nn
from nsrm.graph.hypergraph import RecursiveHypergraph
from nsrm.experts.geometer import ManifoldGeometer
from nsrm.experts.optician import ManifoldOptician
from nsrm.experts.acoustic import ManifoldAcoustic
from nsrm.senses.vision_encoder import ManifoldVisionEncoder
from nsrm.bridge.router import SemanticRouter

class NSRM_Omni(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- The Brain ---
        self.logic_core = RecursiveHypergraph(node_dim=64)
        self.router = SemanticRouter(graph_dim=64, expert_latent_dim=16, num_experts=4)
        
        # --- The Senses (Inputs) ---
        self.eye = ManifoldVisionEncoder(latent_dim=64) # Maps to Graph Dim directly
        
        # --- The Body (Outputs) ---
        self.expert_geometer = ManifoldGeometer(latent_dim=16)
        self.expert_optician = ManifoldOptician(latent_dim=16)
        self.expert_acoustic = ManifoldAcoustic(latent_dim=16)
        
    def perceive(self, image):
        """
        Step 1: Look.
        Takes an image and converts it into a Graph State.
        """
        visual_concept = self.eye(image) # (B, 64)
        
        # Inject into the Brain (The image becomes a thought)
        graph_state = self.logic_core(visual_concept)
        return graph_state

    def imagine(self, graph_state, modality='3d', coords=None):
        """
        Step 2: Act.
        Uses the current thought to generate reality.
        """
        weights, thought_vector = self.router(graph_state)
        
        if modality == '3d':
            return self.expert_geometer(coords, thought_vector)
        elif modality == 'image':
            return self.expert_optician(coords, thought_vector)
        elif modality == 'audio':
            return self.expert_acoustic(coords, thought_vector)
