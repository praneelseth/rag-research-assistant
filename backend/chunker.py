# backend/chunker.py

from typing import List

def chunk_text(text: str, max_tokens: int = 500, overlap: int = 20) -> List[str]:
    """
    Splits raw text into chunks of ~max_tokens (~0.75 token/word; so 666 words per 500 tokens).
    Uses simple word splitting, with overlap.
    """
    words = text.split()
    approx_words = int(max_tokens / 0.75)
    approx_overlap = int(overlap / 0.75)
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+approx_words]
        chunks.append(" ".join(chunk))
        if i + approx_words >= len(words):
            break
        i += approx_words - approx_overlap
    return chunks