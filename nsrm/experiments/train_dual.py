import torch
import torch.optim as optim
from nsrm.model.nsrm_dual import NSRM_Dual_Mind
from nsrm.loss.physics_loss import eikonal_loss

def train_dual_mind():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NSRM_Dual_Mind().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # --- Concepts ---
    # We define concepts and their EXPECTED expert
    # Sphere -> Expert 0 (Geometer)
    c_sphere = torch.tensor([[1.0] + [0.0]*63]).to(device)
    target_router_sphere = torch.tensor([[1.0, 0.0]]).to(device) 
    
    # Sunset -> Expert 1 (Optician)
    c_sunset = torch.tensor([[0.0] + [1.0] + [0.0]*62]).to(device)
    target_router_sunset = torch.tensor([[0.0, 1.0]]).to(device)
    
    print("Starting Dual-Mind Training (Switching between 3D and 2D)...")
    
    for epoch in range(1501):
        # --- 1. SMART SAMPLING (The Fix) ---
        # Instead of just random points, we mix in points near the surface.
        
        # Batch size split: 50% Uniform, 50% Near Surface
        n_points = 2048
        half = n_points // 2
        
        # Uniform points (to learn empty space)
        uniform_points = (torch.rand(1, half, 3).to(device) * 2 - 1)
        
        # Surface points (to learn the shape details)
        # We cheat slightly during training by perturbing the known target surface
        # For a Sphere: Random directions * radius (0.5) + small noise
        dirs = torch.randn(1, half, 3).to(device)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        surface_points_sphere = (dirs * 0.5) + (torch.randn_like(dirs) * 0.05) # +/- 0.05 noise
        
        # Combine for Sphere Task
        coords_3d_sphere = torch.cat([uniform_points, surface_points_sphere], dim=1).requires_grad_(True)
        
        # --- Task A: Train Sphere (3D) ---
        out_sphere = model(c_sphere, coords_3d=coords_3d_sphere, coords_2d=None)
        
        # Task A Losses:
        # 1. Router Loss
        loss_r_sphere = torch.nn.functional.mse_loss(out_sphere['router_weights'], target_router_sphere)
        
        # 2. Physics Loss: Is it a sphere?
        target_sdf = torch.norm(coords_3d_sphere, dim=-1, keepdim=True) - 0.5
        # Loss Modification: Increase Eikonal weight to 0.5 for smoothness
        loss_geo = torch.abs(out_sphere['sdf'] - target_sdf).mean() + 0.5 * eikonal_loss(out_sphere['sdf'], coords_3d_sphere)
        
        
        # --- Task B: Train Sunset (2D) ---
        coords_2d = (torch.rand(1, 1024, 2).to(device) * 2 - 1).requires_grad_(True)
        out_sunset = model(c_sunset, coords_3d=None, coords_2d=coords_2d)
        
        # Task B Losses:
        # 1. Router Loss: Did it choose Optician?
        loss_r_sunset = torch.nn.functional.mse_loss(out_sunset['router_weights'], target_router_sunset)
        
        # 2. Image Loss: Is it a gradient? (Simple Red->Yellow gradient)
        # 2D coords range [-1, 1]
        y_vals = coords_2d[:, :, 1:2]
        # Target: Red=1, Green=(y+1)/2, Blue=0
        target_img = torch.cat([torch.ones_like(y_vals), (y_vals+1)/2, torch.zeros_like(y_vals)], dim=-1)
        loss_opt = torch.abs(out_sunset['image'] - target_img).mean()
        
        # --- Optimization ---
        total_loss = loss_r_sphere + loss_geo + loss_r_sunset + loss_opt
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss={total_loss.item():.5f} | Router Sphere: {out_sphere['router_weights'][0].tolist()}")

    torch.save(model.state_dict(), "nsrm_dual.pth")
    print("Dual Mind Trained.")

if __name__ == "__main__":
    train_dual_mind()
