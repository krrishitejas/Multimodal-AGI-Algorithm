import torch
import numpy as np
import skimage.measure
import trimesh
from nsrm.model.nsrm_integrated import NSRM_Integrated

def extract_mesh_integrated(model, concept_vector, file_name="output.obj", resolution=64, threshold=0.0):
    model.eval()
    
    # 1. Create a grid of points
    voxel_coords = np.linspace(-1, 1, resolution)
    x, y, z = np.meshgrid(voxel_coords, voxel_coords, voxel_coords, indexing='ij')
    coords = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    
    # 2. Query the Integrated Model
    chunk_size = 4096 * 4
    sdf_values = []
    
    with torch.no_grad():
        coords_tensor = torch.FloatTensor(coords).unsqueeze(0) # (1, N, 3)
        concept_tensor = concept_vector # (1, 64)
        
        for i in range(0, coords.shape[0], chunk_size):
            chunk = coords_tensor[:, i:i+chunk_size, :]
            if torch.cuda.is_available():
                chunk = chunk.cuda()
                concept_tensor = concept_tensor.cuda()
                model = model.cuda()
            
            # The Integrated model returns (sdf, rgb, weights)
            sdf_chunk, _, _ = model(concept_tensor, chunk)
            sdf_values.append(sdf_chunk.cpu().numpy())
            
    sdf_grid = np.concatenate(sdf_values, axis=1).reshape(resolution, resolution, resolution)
    
    # 3. Marching Cubes
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(sdf_grid, level=threshold)
        
        # 4. Save to OBJ
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        mesh.export(file_name)
        print(f"Mesh extracted to '{file_name}' with {len(verts)} vertices.")
    except Exception as e:
        print(f"Failed to extract mesh for {file_name}: {e}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NSRM_Integrated()
    
    try:
        model.load_state_dict(torch.load("nsrm_integrated.pth", map_location=device))
        print("Loaded nsrm_integrated.pth")
    except FileNotFoundError:
        print("Model file 'nsrm_integrated.pth' not found. Please run training first.")
        exit()
    
    model.to(device)
    
    # Define Concepts
    concept_sphere = torch.zeros(1, 64).to(device)
    concept_sphere[0, 0] = 1.0
    
    concept_cube = torch.zeros(1, 64).to(device)
    concept_cube[0, 1] = 1.0
    
    print("Extracting Sphere...")
    extract_mesh_integrated(model, concept_sphere, "nsrm_output_sphere.obj")
    
    print("Extracting Cube...")
    extract_mesh_integrated(model, concept_cube, "nsrm_output_cube.obj")
