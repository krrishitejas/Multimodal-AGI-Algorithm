import torch
import torch.optim as optim
from nsrm.experts.geometer import ManifoldGeometer
from nsrm.loss.physics_loss import eikonal_loss

def train_sphere_discovery():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ManifoldGeometer(latent_dim=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # A fixed "concept" vector for "Sphere"
    sphere_concept = torch.randn(1, 16).to(device)
    
    print("Starting Geometer Training (Target: Unit Sphere)...")
    
    for epoch in range(1001):
        # 1. Sampling Strategy: "The Void"
        # We sample random points in a 3D box [-1, 1]
        # (B, N, 3)
        coords = (torch.rand(1, 4096, 3).to(device) * 2 - 1).requires_grad_(True)
        
        # 2. Forward Pass
        sdf_pred, _ = model(coords, sphere_concept)
        
        # 3. Compute Losses
        
        # A. Reconstruction Loss (The Goal)
        # Target SDF for a unit sphere: distance from center - radius (0.5)
        # Mathematically: ||x||_2 - 0.5
        target_sdf = torch.norm(coords, dim=-1, keepdim=True) - 0.5
        recon_loss = torch.abs(sdf_pred - target_sdf).mean()
        
        # B. Physics Loss (The Constraint)
        # Forces the field to be a valid Signed Distance Function
        phys_loss = eikonal_loss(sdf_pred, coords)
        
        # Total Loss
        loss = recon_loss + (0.1 * phys_loss)
        
        # 4. Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Total Loss={loss.item():.5f} (Phys={phys_loss.item():.5f})")

    # Save the expert
    torch.save(model.state_dict(), "geometer_sphere.pth")
    print("Training Complete. Expert Saved.")

if __name__ == "__main__":
    train_sphere_discovery()
