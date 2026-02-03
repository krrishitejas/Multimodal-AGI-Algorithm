import torch
from torch.utils.data import DataLoader
from nsrm.model.nsrm_all_in_one import NSRM_Omni
from nsrm.bridge.tokenizer import ConceptTokenizer
from nsrm.data.datasets import RealWorldImages
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def train_fine_tune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Initialize the AGI
    # Ensure NSRM_Omni can accept the latent dim from tokenizer if it differs,
    # or ensure tokenizer outputs what NSRM expects.
    # LogicCore expects node_dim=64 by default.
    # Tokenizer adapter outputs latent_dim=64 to match.
    model = NSRM_Omni().to(device)
    tokenizer = ConceptTokenizer(latent_dim=64).to(device)
    
    # 2. Load Reality
    print("Loading dataset...")
    dataset = RealWorldImages()
    # Using 0 workers for safety on some setups, can increase if needed
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # Optimize both the model body and the tokenizer adapter
    optimizer = torch.optim.Adam(list(model.parameters()) + list(tokenizer.parameters()), lr=1e-4)
    
    print("Starting Fine-Tuning on Real World Data...")
    
    # CIFAR-10 classes
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    for epoch in range(10): # Running for 10 epochs as requested
        for batch_idx, (coords, rgb_targets, labels) in enumerate(loader):
            coords = coords.to(device)
            rgb_targets = rgb_targets.to(device)
            
            # --- THE "TEACHING" PROCESS ---
            
            # A. Get text description of the object (e.g., "airplane", "frog")
            text_prompts = [f"A photo of a {class_names[l]}" for l in labels]
            
            # B. Tokenize intent (Text -> Concept Vector)
            intent_vector = tokenizer(text_prompts) # (Batch, 64)
            
            # C. Forward Pass (Brain -> Optician -> Image)
            # The model tries to generate the image described by the text
            # We inject the "intent" into the brain.
            # NSRM_Omni.logic_core expects (Batch, Node_Dim).
            graph_state = model.logic_core(intent_vector) 
            
            # The Optician tries to paint pixels at 'coords'
            # Note: We manually route to Optician for this training phase to force visual learning
            _, thought_vector = model.router(graph_state)
            pred_rgb = model.expert_optician(coords, thought_vector)
            
            # D. Loss (Did it paint the frog correctly?)
            loss = torch.nn.functional.mse_loss(pred_rgb, rgb_targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                
        # Save check point after each epoch
        torch.save(model.state_dict(), f"nsrm_finetuned_epoch_{epoch}.pth")

    torch.save(model.state_dict(), "nsrm_finetuned.pth")
    print("Training Complete. Saved 'nsrm_finetuned.pth'.")

if __name__ == "__main__":
    train_fine_tune()
