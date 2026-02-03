import torch

class Node:
    """
    Represents a node in the Recursive Hypergraph.
    Can be an atomic concept (with an embedding) or a hyperedge (containing other nodes).
    """
    def __init__(self, id, embedding=None, children=None):
        self.id = id
        self.embedding = embedding # Tensor (e.g., from the CSP layer)
        self.children = children if children else [] # List of Node IDs if this is a hyperedge

class RecursiveHypergraph:
    """
    Manages the graph topology.
    """
    def __init__(self):
        self.nodes = {} # id -> Node
        self.next_id = 0
        
    def add_node(self, embedding=None):
        """Create an atomic node with a semantic embedding."""
        node = Node(self.next_id, embedding=embedding)
        self.nodes[self.next_id] = node
        self.next_id += 1
        return node.id
        
    def add_hyperedge(self, child_ids):
        """
        Create a hyperedge comprising a list of existing nodes.
        This represents a compositional concept (e.g., a sentence made of words).
        """
        # Validate children exist
        for cid in child_ids:
            if cid not in self.nodes:
                raise ValueError(f"Node {cid} does not exist")
                
        node = Node(self.next_id, children=child_ids)
        self.nodes[self.next_id] = node
        self.next_id += 1
        return node.id
        
    def get_embedding(self, node_id):
        """
        Retrieve embedding. If node is a hyperedge, aggregate children embeddings recursively.
        """
        if node_id not in self.nodes:
            return None
            
        node = self.nodes[node_id]
        if node.embedding is not None:
            return node.embedding
        elif node.children:
            # Recursive aggregation
            # In a full NSRM, this would be a learned composition function (e.g., GNN allocation).
            # Here we use mean pooling for the PoC.
            child_embs = [self.get_embedding(c) for c in node.children]
            
            # Filter None
            valid_embs = [e for e in child_embs if e is not None]
            
            if valid_embs:
                return torch.stack(valid_embs).mean(dim=0)
                
        return None
