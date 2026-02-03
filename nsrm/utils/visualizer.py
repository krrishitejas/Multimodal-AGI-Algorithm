import torch
import numpy as np
import skimage.measure
import trimesh
from nsrm.experts.geometer import ManifoldGeometer

def extract_mesh(model, latent_code, resolution=64, threshold=0.0):
    model.eval()
    
    # 1. Create a grid of points
    voxel_coords = np.linspace(-1, 1, resolution)
    x, y, z = np.meshgrid(voxel_coords, voxel_coords, voxel_coords, indexing='ij')
    coords = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    
    # 2. Query the Geometer Expert
    # Process in chunks to avoid memory overflow
    chunk_size = 4096 * 4
    sdf_values = []
    
    with torch.no_grad():
        coords_tensor = torch.FloatTensor(coords).unsqueeze(0) # Batch dim
        latent_tensor = latent_code.unsqueeze(0)
        
        for i in range(0, coords.shape[0], chunk_size):
            chunk = coords_tensor[:, i:i+chunk_size, :]
            if torch.cuda.is_available():
                chunk = chunk.cuda()
                latent_tensor = latent_tensor.cuda()
                model = model.cuda()
            
            sdf_chunk, _ = model(chunk, latent_tensor)
            sdf_values.append(sdf_chunk.cpu().numpy())
            
    sdf_grid = np.concatenate(sdf_values, axis=1).reshape(resolution, resolution, resolution)
    
    # 3. Marching Cubes (SDF -> Mesh)
    # Extracts the surface where value is 0
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(sdf_grid, level=threshold)
        
        # 4. Save to OBJ
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        mesh.export('nsrm_output.obj')
        print(f"Mesh extracted to 'nsrm_output.obj' with {len(verts)} vertices.")
    except Exception as e:
        print(f"Failed to extract mesh (level set not found?): {e}")

if __name__ == "__main__":
    # Load model and run extraction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ManifoldGeometer(latent_dim=16)
    
    try:
        model.load_state_dict(torch.load("geometer_sphere.pth", map_location=device))
        print("Loaded geometer_sphere.pth")
    except FileNotFoundError:
        print("Model file 'geometer_sphere.pth' not found. Please run training first.")
        exit()
    
    model.to(device)
    
    # Use the same random latent vector used in training (or a new one if testing generalization)
    # Ideally, save the vector during training to reload it here.
    # For this demo, we assume the model learned the sphere regardless of the vector if overfitted,
    # or you re-generate a random one to see what it does.
    dummy_code = torch.randn(16).to(device)
    
    extract_mesh(model, dummy_code)
