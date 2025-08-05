# backend/vector_db.py

import numpy as np

def create_index(embeddings: np.ndarray):
    """
    Store L2-normalized embeddings as the index matrix.
    """
    emb = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norms, 1e-8, np.inf)
    return emb

def search(index: np.ndarray, query_emb: np.ndarray, top_k: int = 5):
    """
    Compute cosine similarity: index @ query_emb, return indices of top_k.
    """
    q = query_emb.astype(np.float32).reshape(-1)
    q = q / (np.linalg.norm(q) + 1e-8)
    scores = index @ q
    if len(scores) == 0:
        return []
    # Get indices of top_k highest scores
    top_k = min(top_k, len(scores))
    idxs = np.argpartition(-scores, range(top_k))[:top_k]
    # Sort by descending score
    idxs = idxs[np.argsort(-scores[idxs])]
    return idxs.tolist()

def add_embeddings(index: np.ndarray, new_embs: np.ndarray):
    """
    Add new L2-normalized embeddings (row-wise) to the index matrix.
    Returns updated matrix.
    """
    new_embs = np.array(new_embs, dtype=np.float32)
    norms = np.linalg.norm(new_embs, axis=1, keepdims=True)
    new_embs = new_embs / np.clip(norms, 1e-8, np.inf)
    if index is None or len(index) == 0:
        return new_embs
    return np.vstack([index, new_embs])