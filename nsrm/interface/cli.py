import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from nsrm.model.nsrm_integrated import NSRM_Integrated
from nsrm.utils.visualizer_integrated import extract_mesh_integrated

class NSRM_CLI:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing NSRM System on {self.device}...")
        
        # Load Brain
        self.model = NSRM_Integrated().to(self.device)
        self.model_path = "nsrm_integrated.pth"
        
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print("Loaded 'nsrm_integrated.pth'. Brain is active.")
        else:
            print("Warning: Model weights not found. Running with untrained random weights.")
            
        self.model.eval()
        
        # Vocabulary / Concept Map
        # In a full system, this would be the TextToSignal Projector.
        # For now, we manually map keywords to our training vectors.
        self.vocab = {
            "sphere": 0,
            "ball": 0,
            "orb": 0,
            "cube": 1,
            "block": 1,
            "box": 1
        }

    def process_command(self, text):
        text = text.lower().strip()
        
        if text in ["exit", "quit"]:
            return False
            
        print(f"\n[User]: {text}")
        
        # 1. Understand (Naïve Parser)
        concept_idx = -1
        for word, idx in self.vocab.items():
            if word in text:
                concept_idx = idx
                print(f"[NSRM Brain]: Detected concept '{word}' (ID: {idx})")
                break
        
        if concept_idx == -1:
            # Check for Compositional Concepts
            if "human" in text or "person" in text or "man" in text:
                print("[NSRM Brain]: Detected High-Level Concept 'Human'. Decomposing...")
                print(" -> Component 1: Sphere (Head) at y=+0.6")
                print(" -> Component 2: Cube (Body) at y=-0.4")
                
                filename = f"chat_output_{text.replace(' ', '_')}.obj"
                print(f"[NSRM Body]: assembling composite geometry -> {filename}...")
                
                # We need a new composite extractor
                self.generate_composite_human(filename)
                print(f"[NSRM]: Done. Artificial Human created.")
                return True

            print("[NSRM Brain]: I don't understand that concept yet. Try 'sphere', 'cube', or 'human'.")
            return True
            
        # 2. Transduce (Create Thought Vector)
        concept_vector = torch.zeros(1, 64).to(self.device)
        concept_vector[0, concept_idx] = 1.0
        
        # 3. Simulate Forward Pass to get Router decision (just for display)
        with torch.no_grad():
            weights, _ = self.model.router(concept_vector)
            geometer_confidence = weights[0, 0].item()
            print(f"[NSRM Nervous System]: Routing to Geometer Expert (Confidence: {geometer_confidence:.2%})")
        
        # 4. Act (Generate Geometry)
        geometer_weight = weights[0, 0].item()
        
        if geometer_weight > 0.1: # Lower threshold as router is soft-gated
             filename = f"chat_output_{text.replace(' ', '_')}.obj"
             print(f"[NSRM Body]: Generating 3D geometry -> {filename}...")
             extract_mesh_integrated(self.model, concept_vector, filename, resolution=64)
             print(f"[NSRM]: Done. Object created.")
        else:
            print("[NSRM]: Confidence too low to generate architecture.")
            
        return True

    def generate_composite_human(self, filename, resolution=64):
        """
        Manually constructing a 'Human' from primitive experts.
        Head = Sphere (Concept 0)
        Body = Cube (Concept 1)
        Union = min(SDF1, SDF2)
        """
        import numpy as np
        import skimage.measure
        import trimesh
        
        # 1. Grid
        voxel_coords = np.linspace(-1, 1, resolution)
        x, y, z = np.meshgrid(voxel_coords, voxel_coords, voxel_coords, indexing='ij')
        coords = np.stack([x, y, z], axis=-1).reshape(-1, 3) # (N, 3)
        
        # 2. Concepts
        sphere_concept = torch.zeros(1, 64).to(self.device); sphere_concept[0,0] = 1.0
        cube_concept = torch.zeros(1, 64).to(self.device); cube_concept[0,1] = 1.0
        
        chunk_size = 4096 * 4
        sdf_values = []
        
        with torch.no_grad():
            coords_tensor = torch.FloatTensor(coords).unsqueeze(0).to(self.device) # (1, N, 3)
            
            for i in range(0, coords.shape[0], chunk_size):
                chunk = coords_tensor[:, i:i+chunk_size, :]
                
                # -- Head (Sphere) --
                # Shift coordinates DOWN to move object UP
                # y' = y - 0.6
                chunk_head = chunk.clone()
                chunk_head[:, :, 1] -= 0.6
                # Scale head down? Let's say head is smaller.
                # SDF(x/s)*s. 
                # For simplicity, we just use the base sphere (r=0.5).
                
                sdf_head, _, _ = self.model(sphere_concept, chunk_head)
                
                # -- Body (Cube) --
                # Shift coordinates UP to move object DOWN
                # y' = y + 0.5
                chunk_body = chunk.clone()
                chunk_body[:, :, 1] += 0.5
                
                sdf_body, _, _ = self.model(cube_concept, chunk_body)
                
                # -- Union --
                # Soft min or hard min
                sdf_union = torch.min(sdf_head, sdf_body)
                
                sdf_values.append(sdf_union.cpu().numpy())
                
        sdf_grid = np.concatenate(sdf_values, axis=1).reshape(resolution, resolution, resolution)
        
        try:
            verts, faces, normals, values = skimage.measure.marching_cubes(sdf_grid, level=0.0)
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
            mesh.export(filename)
            print(f"Mesh extracted with {len(verts)} vertices.")
        except Exception as e:
            print(f"Failed to mesh human: {e}")

    def run(self):
        print("\n--- NSRM Terminal Interface ---")
        print("Talk to the Neuro-Symbolic Brain. Type 'exit' to stop.")
        
        while True:
            try:
                user_input = input(">> ")
                if not self.process_command(user_input):
                    break
            except KeyboardInterrupt:
                break
        print("\n[NSRM]: Shutting down.")

if __name__ == "__main__":
    cli = NSRM_CLI()
    cli.run()
