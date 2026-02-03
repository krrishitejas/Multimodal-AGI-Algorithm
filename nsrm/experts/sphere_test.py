import torch
import torch.optim as optim
import numpy as np
from nsrm.experts.geometer import GeometerExpert
from nsrm.loss.physics_loss import eikonal_loss

def true_sphere_sdf(coords, radius=1.0):
    """Ground truth SDF for a sphere at origin."""
    return coords.norm(2, dim=1, keepdim=True) - radius

def run_sphere_test():
    print("Running Geometer Expert Sphere Test...")
    
    model = GeometerExpert()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print("Training Geometer to be a Sphere (Radius=1.0)...")
    
    for step in range(1001):
        # Sample random points in 3D space [-2, 2]
        coords = (torch.rand(1024, 3) * 4.0 - 2.0).requires_grad_(True)
        
        # 1. Forward Pass
        pred_sdf, _ = model(coords)
        
        # 2. Compute Losses
        
        # Ground Truth Supervision (Reconstruction Loss)
        # In a real "generative" scenario, we wouldn't have this dense GT.
        # We would probably have a sparse set of points or logical constraints.
        # But here we are testing "Can it learn?", so we use GT.
        gt_sdf = true_sphere_sdf(coords)
        recon_loss = torch.abs(pred_sdf - gt_sdf).mean()
        
        # Physics Loss (Eikonal)
        # Forces proper gradient field
        phys_loss = eikonal_loss(pred_sdf, coords)
        
        total_loss = recon_loss + 0.1 * phys_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step}: Recon Loss = {recon_loss.item():.6f}, Eikonal Loss = {phys_loss.item():.6f}")

    # Validation
    print("\nValidation on X-axis scan:")
    test_x = torch.linspace(-1.5, 1.5, 10).view(-1, 1)
    test_coords = torch.zeros(10, 3)
    test_coords[:, 0] = test_x[:, 0] # (x, 0, 0)
    
    with torch.no_grad():
        pred, _ = model(test_coords)
        gt = true_sphere_sdf(test_coords)
    
    print(f"{'X':>6} | {'Pred':>8} | {'GT':>8} | {'Diff':>8}")
    for i in range(10):
        print(f"{test_coords[i,0].item():6.2f} | {pred[i,0].item():8.4f} | {gt[i,0].item():8.4f} | {abs(pred[i,0]-gt[i,0]).item():8.4f}")
        
    print("\nSphere Test Complete.")

if __name__ == "__main__":
    run_sphere_test()
