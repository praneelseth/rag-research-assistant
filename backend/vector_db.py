# backend/vector_db.py

import faiss
import numpy as np

def create_index(embeddings: np.ndarray):
    """
    Create a FAISS IndexFlatIP (for cosine similarity).
    """
    # Ensure embeddings are L2 normalized
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def search(index, query_emb: np.ndarray, top_k: int = 5):
    """
    Search index with 1D query_emb, returns indices of top_k results.
    """
    q = query_emb.astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(q)
    D, I = index.search(q, top_k)
    return I[0].tolist()

def add_embeddings(index, new_embs):
    """
    Add new L2-normalized embeddings to an existing index.
    """
    faiss.normalize_L2(new_embs)
    index.add(new_embs)