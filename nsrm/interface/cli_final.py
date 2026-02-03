import torch
import numpy as np
import trimesh
import skimage.measure
import matplotlib.pyplot as plt
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from nsrm.model.nsrm_dual import NSRM_Dual_Mind
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mapping Concepts to Training Vectors
CONCEPTS = {
    "sphere": [1.0] + [0.0]*63,  # Maps to 3D
    "sunset": [0.0] + [1.0] + [0.0]*62 # Maps to 2D
}

def render_3d(model, concept_vec):
    # Marching Cubes Logic
    res = 64
    x = np.linspace(-1, 1, res)
    grid = np.stack(np.meshgrid(x, x, x, indexing='ij'), axis=-1).reshape(-1, 3)
    
    with torch.no_grad():
        coords = torch.FloatTensor(grid).unsqueeze(0).to(DEVICE)
        # Reshape to (1, N, 3)
        coords = coords.view(1, -1, 3)
        concept = torch.FloatTensor(concept_vec).unsqueeze(0).to(DEVICE)
        out = model(concept, coords_3d=coords)
        
    sdf = out['sdf'].cpu().numpy().reshape(res, res, res)
    
    try:
        verts, faces, _, _ = skimage.measure.marching_cubes(sdf, level=0.0)
        trimesh.Trimesh(verts, faces).export("output_3d.obj")
        print(" -> [Geometer] 3D Model saved to 'output_3d.obj'")
    except ValueError:
        print(" -> [Geometer] Failed to construct surface (level=0.0 not found).")
    except Exception as e:
        print(f" -> [Geometer] Error: {e}")

def render_2d(model, concept_vec):
    # Pixel Grid Logic
    res = 128
    x = np.linspace(-1, 1, res)
    grid = np.stack(np.meshgrid(x, x, indexing='ij'), axis=-1).reshape(-1, 2)
    
    with torch.no_grad():
        coords = torch.FloatTensor(grid).unsqueeze(0).to(DEVICE)
        # Reshape to (1, N, 2)
        coords = coords.view(1, -1, 2)
        
        concept = torch.FloatTensor(concept_vec).unsqueeze(0).to(DEVICE)
        out = model(concept, coords_2d=coords)
    
    img = out['image'].view(res, res, 3).cpu().numpy()
    plt.imsave("output_2d.png", img)
    print(" -> [Optician] Image saved to 'output_2d.png'")

def main():
    print("Loading Dual-Lobe NSRM...")
    model = NSRM_Dual_Mind().to(DEVICE)
    try:
        model.load_state_dict(torch.load("nsrm_dual.pth", map_location=DEVICE))
        print("Model loaded successfully.")
    except Exception as e:
        print("Warning: Could not load trained weights. Running random.")
        
    model.eval()
    
    print("\n--- NSRM Dual-Mind CLI ---")
    print("Commands: 'sphere', 'sunset', 'exit'")
    
    while True:
        try:
            cmd = input("ATEON-NSRM (Dual) >> ").lower().strip()
        except KeyboardInterrupt:
            break
            
        if cmd == "exit": break
        
        vec = None
        for k in CONCEPTS:
            if k in cmd: vec = CONCEPTS[k]
        
        if vec:
            # 1. Ask Router "What is this?"
            with torch.no_grad():
                concept_t = torch.FloatTensor(vec).unsqueeze(0).to(DEVICE)
                # We do a dummy pass to get weights via forward, or just call router
                # Using model.router directly is cleaner
                weights, _ = model.router(concept_t)
                w = weights[0].cpu().numpy()
            
            print(f" -> Intent Analysis: 3D={w[0]:.2f} | 2D={w[1]:.2f}")
            
            # 2. Route Execution
            if w[0] > w[1]:
                print(" -> Routing to Geometer...")
                render_3d(model, vec)
            else:
                print(" -> Routing to Optician...")
                render_2d(model, vec)
        else:
            print(" -> Concept not trained yet. Try 'sphere' or 'sunset'.")

if __name__ == "__main__":
    main()
