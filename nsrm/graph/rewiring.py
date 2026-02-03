import torch

def compute_spectral_correlation(embeddings):
    """
    Computes pairwise similarity between embeddings.
    embeddings: (N, D)
    Returns: (N, N) correlation matrix
    """
    # Normalize for cosine similarity
    # In NSRM, this might be correlation in the frequency domain (Spectral Resonance)
    # Cosine similarity in standard domain is equivalent if normalized.
    if embeddings.size(0) == 0:
        return torch.tensor([])
        
    norm_emb = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    similarity = torch.mm(norm_emb, norm_emb.t())
    return similarity

def propose_rewiring(graph, threshold=0.8):
    """
    Analyzes the graph's current embeddings and proposes new hyperedges
    based on spectral resonance (high correlation).
    """
    # 1. Collect all valid embeddings currently in the graph
    ids = []
    embs = []
    for nid in graph.nodes:
        e = graph.get_embedding(nid)
        if e is not None:
            ids.append(nid)
            embs.append(e)
            
    if not embs:
        return []
        
    embs_tensor = torch.stack(embs) # (N, D)
    
    # 2. Compute Resonance
    sim = compute_spectral_correlation(embs_tensor)
    
    # 3. Find resonating pairs
    N = len(ids)
    new_edges = []
    
    # Avoid self-loops and duplicates
    for i in range(N):
        for j in range(i + 1, N):
            if sim[i, j] > threshold:
                # Propose connection
                # Return tuple of IDs that should be bound
                new_edges.append([ids[i], ids[j]])
    
    return new_edges
