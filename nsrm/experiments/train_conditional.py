import torch
import torch.optim as optim
from nsrm.model.nsrm_integrated import NSRM_Integrated
from nsrm.loss.physics_loss import eikonal_loss

def train_conditional_creation():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NSRM_Integrated().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Define "Concepts" (simulating inputs from the Text Projector)
    # We use 64-dim vectors to represent "Sphere" and "Cube"
    # One-hot-like or distinct vectors
    concept_sphere = torch.zeros(1, 64).to(device)
    concept_sphere[0, 0] = 1.0 # Index 0 is Sphere
    
    concept_cube = torch.zeros(1, 64).to(device)
    concept_cube[0, 1] = 1.0 # Index 1 is Cube
    
    print("Starting Conditional Training (Sphere vs. Cube)...")
    
    # Increase training duration for convergence
    for epoch in range(6001):
        # 1. Sample Space
        # (B, N, 3)
        coords = (torch.rand(1, 2048, 3).to(device) * 2 - 1).requires_grad_(True)
        
        # --- Task A: Sphere ---
        sdf_sphere, _, _ = model(concept_sphere, coords)
        # Target: Sphere Radius 0.5
        target_sphere = torch.norm(coords, dim=-1, keepdim=True) - 0.5
        # Stronger Eikonal (0.5) to force smooth, unit-gradient field
        loss_sphere = torch.abs(sdf_sphere - target_sphere).mean() + \
                      0.5 * eikonal_loss(sdf_sphere, coords)
        
        # --- Task B: Cube ---
        sdf_cube, _, _ = model(concept_cube, coords)
        # Math for Cube SDF: max(|x|, |y|, |z|) - radius
        # Radius 0.5
        target_cube = torch.amax(torch.abs(coords), dim=-1, keepdim=True) - 0.5
        loss_cube = torch.abs(sdf_cube - target_cube).mean() + \
                    0.5 * eikonal_loss(sdf_cube, coords)
        
        # Combined Loss
        total_loss = loss_sphere + loss_cube
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss {total_loss.item():.5f}")

    torch.save(model.state_dict(), "nsrm_integrated.pth")
    print("Integration Complete.")

if __name__ == "__main__":
    train_conditional_creation()
