import torch
import unittest
from nsrm.csp.inr import INRNetwork
from nsrm.csp.fno import FNOBlock
from nsrm.spectral.gfnet import GlobalFilterLayer
from nsrm.model.nsrm_agl import NSRMModel
from nsrm.graph.hypergraph import RecursiveHypergraph

class TestNSRM(unittest.TestCase):
    def test_inr(self):
        print("\nTesting INR...")
        model = INRNetwork(input_dim=2, hidden_dim=64, output_dim=1)
        x = torch.randn(10, 2) # Batch of 10 points
        y = model(x)
        self.assertEqual(y.shape, (10, 1))
        print("INR Pass: OK")

    def test_fno_block(self):
        print("\nTesting FNO Block...")
        # (B, Channels, L)
        x = torch.randn(2, 32, 128) 
        model = FNOBlock(width=32, modes=16)
        y = model(x)
        self.assertEqual(y.shape, (2, 32, 128))
        print("FNO Block Pass: OK")

    def test_gfnet(self):
        print("\nTesting GFNet...")
        x = torch.randn(2, 32, 128)
        model = GlobalFilterLayer(dim=32, sequence_length=128)
        y = model(x) # Expect (B, C, L)
        self.assertEqual(y.shape, (2, 32, 128))
        print("GFNet Pass: OK")

    def test_nsrm_model(self):
        print("\nTesting Full NSRM Model...")
        # (B, L, D_in)
        x = torch.randn(2, 128, 10)
        model = NSRMModel(input_dim=10, model_dim=64, output_dim=5, seq_len=128)
        y = model(x)
        self.assertEqual(y.shape, (2, 128, 5))
        
        # Test Gradients
        loss = y.sum()
        loss.backward()
        print("NSRM Backward Pass: OK")
        
    def test_graph_rewiring(self):
        print("\nTesting Graph Rewiring...")
        model = NSRMModel(10, 64, 5)
        
        # Add two resonating nodes
        emb1 = torch.tensor([1.0, 0.0])
        emb2 = torch.tensor([1.0, 0.1]) # High cosine sim
        emb3 = torch.tensor([0.0, 1.0]) # Orthogonal
        
        id1 = model.graph.add_node(emb1) 
        id2 = model.graph.add_node(emb2)
        id3 = model.graph.add_node(emb3)
        
        # Manually trigger similarity check
        new_ids = model.symbolic_update({}) # Embeddings already set
        
        # Should connect id1 and id2
        print(f"New Hyperedges Created: {new_ids}")
        # We expect at least one connection between 1 and 2
        # Note: Depending on threshold in rewiring.py (0.8), [1,0] and [1, 0.1] are very close.
        self.assertTrue(len(new_ids) > 0)

if __name__ == '__main__':
    unittest.main()
