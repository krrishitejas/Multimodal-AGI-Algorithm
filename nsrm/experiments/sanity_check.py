import torch
import torch.nn as nn
import torch.optim as optim
from nsrm.model.nsrm_agl import NSRMModel
from nsrm.bridge.projector import TextToSignal
from nsrm.train.trainer import HybridTrainer

def run_experiment():
    print("Running NSRM Sanity Check: '1, 2, 3, 4 -> 5'")
    
    # 1. Setup Models
    # Dimensions: 1D signal for simplicity (just the value)
    model_dim = 16
    input_dim = 1 # We will treat the number as a 1D scalar signal value 
    seq_len = 5
    
    # The NSRM Model
    model = NSRMModel(input_dim=input_dim, model_dim=model_dim, output_dim=1, seq_len=seq_len)
    
    # The Bridge (Siren Projector)
    # Maps Position -> Signal. 
    projector = TextToSignal(embedding_dim=input_dim, hidden_dim=32)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    trainer = HybridTrainer(model, optimizer)
    
    # 2. Data Preparation
    # Sequence: 1, 2, 3, 4
    # Target: 5
    # We treat these as continuous values.
    # Input: (Batch=1, Seq=4, Dim=1) -> [[1], [2], [3], [4]]
    # But wait, NSRM expects fixed length or continuous. Let's say input is simple tensor.
    
    inputs = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]) # (1, 4, 1)
    
    # Target: We want the model to predict the *next* step.
    # Given 1..4, predict 2..5.
    targets = torch.tensor([[[2.0], [3.0], [4.0], [5.0]]])
    
    # 3. Symbolic Context (Mockup)
    # We represent "1", "2", "3", "4" as nodes in the graph
    symbolic_context = {
        0: torch.tensor([1.0] * model_dim), # Node "1"
        1: torch.tensor([2.0] * model_dim), # Node "2"
        2: torch.tensor([3.0] * model_dim), # Node "3" 
        3: torch.tensor([4.0] * model_dim)  # Node "4"
    }
    # We assign them embeddings that are linearly increasing to induce resonance (correlation)
    # 3.1 Initialize Graph
    for nid, emb in symbolic_context.items():
        # In this PoC we manually add them. In real NSRM, an "Encoder" would spawn them.
        model.graph.add_node(emb) # This returns new IDs, we assume they match 0,1,2,3 for simplicity

    
    # 4. Training Loop
    print("\nStarting Training Loop...")
    for epoch in range(50):
        # We need to project inputs? 
        # For this numeric task, raw values are fine. 
        # But let's use projector to add some "positional signal" to inputs
        pos = torch.linspace(0, 1, 4).view(1, 4, 1)
        pos_signal = projector(pos) # (1, 4, 1)
        
        # Combine raw value + pos signal? 
        # Let's just use raw values for the core model to prove FNO learns linear trend.
        
        loss, new_edges = trainer.train_step(inputs, targets, symbolic_context)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f}")
            if new_edges:
                print(f"  -> Graph Updated! New Hyperedges: {new_edges}")

    # 5. Final Evaluation
    print("\nFinal Check:")
    with torch.no_grad():
        final_pred = model(inputs)
        print(f"Input: [1, 2, 3, 4]")
        print(f"Predicted Last Step (Expect ~5.0): {final_pred[0, -1, 0].item():.4f}")
        
    print("\nGraph State:")
    print(f"Total Nodes: {len(model.graph.nodes)}")
    # We expect some rewiring if resonance happened
    
if __name__ == "__main__":
    run_experiment()
