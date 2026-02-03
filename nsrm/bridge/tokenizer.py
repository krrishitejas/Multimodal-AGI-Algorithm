import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel

class ConceptTokenizer(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        # Load a pre-trained "Teacher" (CLIP)
        # using openai/clip-vit-base-patch32 as requested
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # Mapper: Adapts CLIP's 512-dim vector to your 16-dim NSRM format
        self.adapter = nn.Linear(512, latent_dim)
        
        # Freeze CLIP (We only train the adapter)
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, text_list):
        """
        Input: ["A red sphere", "A loud sound"]
        Output: (Batch, 16) -> Concept Vectors
        """
        inputs = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
        
        if torch.cuda.is_available():
            device = next(self.parameters()).device
            inputs = inputs.to(device)
            # Ensure encoder is on the correct device if it wasn't moved already
            self.encoder = self.encoder.to(device)
            
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # Take the [CLS] token representation
            clip_emb = outputs.last_hidden_state[:, 0, :] # (Batch, 512)
            
        # Project to your Manifold Space
        concept_vector = self.adapter(clip_emb.float())
        return concept_vector
