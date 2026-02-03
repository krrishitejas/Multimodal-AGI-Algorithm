import torch
import numpy as np
from nsrm.senses.vision_encoder import ManifoldVisionEncoder

def test_vision():
    # 1. Setup
    eye = ManifoldVisionEncoder(latent_dim=16)
    
    # 2. Create a Synthetic "Sunset" Image (Red/Yellow Gradient)
    # This simulates "Seeing" something real
    res = 64
    x = np.linspace(-1, 1, res)
    grid = np.stack(np.meshgrid(x, x, indexing='ij'), axis=-1)
    y_vals = grid[:, :, 1]
    
    img_data = np.zeros((res, res, 3))
    # Fill manually to avoid shape issues
    img_data[:, :, 0] = 1.0 # Red
    for i in range(res):
        for j in range(res):
            img_data[i, j, 1] = (y_vals[i, j] + 1) / 2 # Green gradient

    
    # Convert to Tensor (Batch, Channels, Height, Width)
    img_tensor = torch.FloatTensor(img_data).permute(2, 0, 1).unsqueeze(0)
    
    # 3. Perceive
    print("Injecting image into Vision Encoder...")
    latent_vector = eye(img_tensor)
    
    print(f"Extraction Complete. Vector Shape: {latent_vector.shape}")
    print(f"Concept Vector Summary: Mean={latent_vector.mean().item():.3f}")
    
    # In a real app, you would now compare this vector to your dictionary:
    # dist_sphere = |vector - sphere_concept|
    # dist_sunset = |vector - sunset_concept|
    # If dist_sunset is lower, the AI says: "I see a sunset."

if __name__ == "__main__":
    test_vision()
