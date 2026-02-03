import torch
import torch.optim as optim
import torch.nn.functional as F
from nsrm.model.nsrm_trinity import NSRM_Trinity_Mind
from nsrm.loss.physics_loss import eikonal_loss

def train_trinity_mind():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NSRM_Trinity_Mind(vocab_size=25).to(device) # Small vocab matches CLI
    optimizer = optim.Adam(model.parameters(), lr=1e-3) # Higher LR for quick routing learning
    
    # --- Concepts ---
    # 1. Sphere (3D) -> Expert 0
    c_sphere = torch.tensor([[1.0] + [0.0]*63]).to(device)
    t_router_sphere = torch.tensor([[1.0, 0.0, 0.0]]).to(device) 
    
    # 2. Sunset (2D) -> Expert 1
    c_sunset = torch.tensor([[0.0] + [1.0] + [0.0]*62]).to(device)
    t_router_sunset = torch.tensor([[0.0, 1.0, 0.0]]).to(device)

    # 3. Poem (Text) -> Expert 2
    c_poem = torch.tensor([[0.0]*2 + [1.0] + [0.0]*61]).to(device)
    t_router_poem = torch.tensor([[0.0, 0.0, 1.0]]).to(device)
    
    print("Starting Trinity Training (Teaching 3-Way Semantic Routing)...")
    
    for epoch in range(1001):
        # Sample Domains
        coords_3d = (torch.rand(1, 512, 3).to(device) * 2 - 1).requires_grad_(True)
        coords_2d = (torch.rand(1, 512, 2).to(device) * 2 - 1).requires_grad_(True)
        # Text seq len
        seq_len = 5 # Reduced to match target
        
        # --- Task A: Sphere (3D) ---
        out_sphere = model(c_sphere, coords_3d=coords_3d)
        loss_r_sphere = F.mse_loss(out_sphere['router_weights'], t_router_sphere)
        # Physics proxy
        target_sdf = torch.norm(coords_3d, dim=-1, keepdim=True) - 0.5
        loss_geo = torch.abs(out_sphere['sdf'] - target_sdf).mean()
        
        # --- Task B: Sunset (2D) ---
        out_sunset = model(c_sunset, coords_2d=coords_2d)
        loss_r_sunset = F.mse_loss(out_sunset['router_weights'], t_router_sunset)
        # Image proxy
        y_vals = coords_2d[:, :, 1:2]
        target_img = torch.cat([torch.ones_like(y_vals), (y_vals+1)/2, torch.zeros_like(y_vals)], dim=-1)
        loss_opt = torch.abs(out_sunset['image'] - target_img).mean()
        
        # --- Task C: Poem (Text) ---
        out_poem = model(c_poem, text_seq_len=seq_len)
        loss_r_poem = F.mse_loss(out_poem['router_weights'], t_router_poem)
        
        # 3. Language Loss: Teach it a specific sentence
        # "manifold logic is continuous flow" (No padding)
        # VOCAB indices: manifold=7, logic=15, is=3, continuous=17, flow=18
        target_seq = torch.tensor([7, 15, 3, 17, 18]).to(device).unsqueeze(0) 
        
        # Logits: (1, 5, 25). Target: (1, 5)
        loss_text = F.cross_entropy(out_poem['text_logits'].permute(0, 2, 1), target_seq)

        # Total Loss
        # Increase text loss weight to 10.0 to force learning
        total_loss = loss_r_sphere + loss_r_sunset + loss_r_poem + loss_geo + loss_opt + (10.0 * loss_text)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Total={total_loss.item():.4f}")
            print(f"  > Lo_R={loss_r_poem.item():.4f} | Lo_Geo={loss_geo.item():.4f} | Lo_Opt={loss_opt.item():.4f} | Lo_Text={loss_text.item():.4f}")
            print(f"  -> Sphere Check: {out_sphere['router_weights'][0].tolist()}")
            print(f"  -> Poem Check:   {out_poem['router_weights'][0].tolist()}")
            
            # Decode sample
            logits = out_poem['text_logits']
            probs = torch.softmax(logits, dim=-1)
            indices = torch.argmax(probs, dim=-1)[0]
            print(f"  -> Text Sample: {indices.tolist()}")

    torch.save(model.state_dict(), "nsrm_trinity.pth")
    print("Trinity Mind Trained.")

if __name__ == "__main__":
    train_trinity_mind()
