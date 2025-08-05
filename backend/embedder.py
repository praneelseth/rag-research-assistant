# backend/embedder.py

from typing import List
import numpy as np

# Singleton for efficient model loading
_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Returns L2-normalized embeddings (n_texts, 384)
    """
    model = get_model()
    embs = model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return embs