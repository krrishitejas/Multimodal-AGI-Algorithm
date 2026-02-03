import torch
import numpy as np
import trimesh
import skimage.measure
import matplotlib.pyplot as plt
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from nsrm.model.nsrm_trinity import NSRM_Trinity_Mind

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Concepts mapping
CONCEPTS = {
    "sphere": [1.0] + [0.0]*63,          # 3D
    "sunset": [0.0] + [1.0] + [0.0]*62,  # 2D
    "poem":   [0.0]*2 + [1.0] + [0.0]*61 # Text
}

# Simple Vocab for prototype
VOCAB = ["hello", "world", "this", "is", "a", "test", "of", "manifold", "linguistics", 
         "sphere", "cube", "sunset", "red", "blue", "green", "logic", "reasoning", 
         "continuous", "flow", "ai", "agi", "thought", "vector", "trajectory", "."]
VOCAB_SIZE = len(VOCAB)

def decode_text(logits):
    # Logits: (1, Seq_Len, Vocab)
    probs = torch.softmax(logits, dim=-1)
    indices = torch.argmax(probs, dim=-1)[0] # (Seq_Len)
    words = [VOCAB[idx.item() % len(VOCAB)] for idx in indices]
    return " ".join(words)

def render_3d(model, concept_vec):
    res = 64
    x = np.linspace(-1, 1, res)
    grid = np.stack(np.meshgrid(x, x, x, indexing='ij'), axis=-1).reshape(-1, 3)
    
    with torch.no_grad():
        coords = torch.FloatTensor(grid).unsqueeze(0).to(DEVICE).view(1, -1, 3)
        concept = torch.FloatTensor(concept_vec).unsqueeze(0).to(DEVICE)
        out = model(concept, coords_3d=coords)
        
    sdf = out['sdf'].cpu().numpy().reshape(res, res, res)
    try:
        verts, faces, _, _ = skimage.measure.marching_cubes(sdf, level=0.0)
        trimesh.Trimesh(verts, faces).export("output_trinity_3d.obj")
        print(" -> [Geometer] 3D Model saved to 'output_trinity_3d.obj'")
    except:
        print(" -> [Geometer] Failed to construct surface.")

def render_2d(model, concept_vec):
    res = 128
    x = np.linspace(-1, 1, res)
    grid = np.stack(np.meshgrid(x, x, indexing='ij'), axis=-1).reshape(-1, 2)
    
    with torch.no_grad():
        coords = torch.FloatTensor(grid).unsqueeze(0).to(DEVICE).view(1, -1, 2)
        concept = torch.FloatTensor(concept_vec).unsqueeze(0).to(DEVICE)
        out = model(concept, coords_2d=coords)
    
    img = out['image'].view(res, res, 3).cpu().numpy()
    plt.imsave("output_trinity_2d.png", img)
    print(" -> [Optician] Image saved to 'output_trinity_2d.png'")

def generate_text(model, concept_vec):
    with torch.no_grad():
        concept = torch.FloatTensor(concept_vec).unsqueeze(0).to(DEVICE)
        out = model(concept, text_seq_len=10)
        
    text = decode_text(out['text_logits'])
    print(f" -> [Linguist] Trajectory Decoded: \"{text}\"")

def main():
    print("Loading NSRM Trinity Mind...")
    model = NSRM_Trinity_Mind(vocab_size=VOCAB_SIZE).to(DEVICE)
    try:
        model.load_state_dict(torch.load("nsrm_trinity.pth", map_location=DEVICE))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Warning: 'nsrm_trinity.pth' not found. Model is untrained (random routing).")
        print("Run 'python -m nsrm.experiments.train_trinity' to train.")
    model.eval()
    
    print("\n--- NSRM Trinity Interface (All-In-One) ---")
    print("Capabilities: Geometer (3D), Optician (2D), Linguist (Text)")
    print("Try commands like: 'sphere', 'sunset', 'write a poem'")
    
    while True:
        try:
            cmd = input("ATEON-NSRM (Trinity) >> ").lower().strip()
        except KeyboardInterrupt:
            break
        if cmd == "exit": break
        
        vec = None
        for k in CONCEPTS:
            if k in cmd: vec = CONCEPTS[k]
        if "write" in cmd or "text" in cmd: vec = CONCEPTS["poem"]

        if vec:
            with torch.no_grad():
                concept_t = torch.FloatTensor(vec).unsqueeze(0).to(DEVICE)
                weights, _ = model.router(concept_t)
                w = weights[0].cpu().numpy() # [Geo, Opt, Lin]
            
            print(f" -> Routing Analysis: 3D={w[0]:.2f} | 2D={w[1]:.2f} | Text={w[2]:.2f}")
            
            best = np.argmax(w)
            if best == 0:
                print(" -> Activating Geometer...")
                render_3d(model, vec)
            elif best == 1:
                print(" -> Activating Optician...")
                render_2d(model, vec)
            else:
                print(" -> Activating Linguist...")
                generate_text(model, vec)
        else:
             print(" -> Concept not recognized in this prototype registry.")

if __name__ == "__main__":
    main()
