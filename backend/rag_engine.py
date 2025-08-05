# backend/rag_engine.py

import os
import threading
from typing import List

# Model URLs (Q4 quantized)
DEFAULT_MODEL_URL = "https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct-GGUF/resolve/main/deepseek-coder-1.3b-instruct.q4_k_m.gguf"
DEFAULT_MODEL_NAME = "deepseek-coder-1.3b-instruct.q4_k_m.gguf"
FALLBACK_URL = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
FALLBACK_NAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

_model = None
_model_id = None
_model_lock = threading.Lock()

def download_if_missing(url, fname):
    """Download model file if not already present."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    outpath = os.path.join(MODELS_DIR, fname)
    if not os.path.isfile(outpath):
        import huggingface_hub
        huggingface_hub.hf_hub_download(
            repo_id="/".join(url.split("/")[-6:-2]),
            filename=fname,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    return outpath

def ensure_model_loaded():
    """
    Loads the main model, or falls back to TinyLlama if OOM.
    Returns the model id.
    """
    global _model, _model_id
    with _model_lock:
        if _model is not None:
            return _model_id
        # Try DeepSeek first
        try:
            model_path = os.path.join(MODELS_DIR, DEFAULT_MODEL_NAME)
            if not os.path.isfile(model_path):
                # Download (stream, avoid RAM spike)
                import requests
                with requests.get(DEFAULT_MODEL_URL, stream=True) as r:
                    r.raise_for_status()
                    with open(model_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            from llama_cpp import Llama
            _model = Llama(model_path=model_path, n_ctx=16384, n_threads=4)
            _model_id = "deepseek-coder-1.3b-instruct.q4"
            return _model_id
        except Exception as e:
            print(f"Failed to load DeepSeek-Coder: {e}")
            # Try fallback TinyLlama
            try:
                fb_path = os.path.join(MODELS_DIR, FALLBACK_NAME)
                if not os.path.isfile(fb_path):
                    import requests
                    with requests.get(FALLBACK_URL, stream=True) as r:
                        r.raise_for_status()
                        with open(fb_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                from llama_cpp import Llama
                _model = Llama(model_path=fb_path, n_ctx=4096, n_threads=4)
                _model_id = "tinyllama-1.1b-chat.q4"
                return _model_id
            except Exception as ee:
                print(f"Failed to load fallback TinyLlama: {ee}")
                raise RuntimeError("Could not load any local Llama model.")
    return _model_id

def get_model_id():
    return _model_id

def answer_question(question: str, chunks: List[str]) -> str:
    """
    Generate answer to question using the Llama model and retrieved context.
    """
    ensure_model_loaded()
    # System prompt template
    system_prompt = (
        "You are a helpful research assistant. "
        "Given the following document excerpts, answer the user's question as accurately and concisely as possible.\n\n"
        "Document context:\n"
    )
    context = "\n\n".join([f"Excerpt {i + 1}:\n{chunk}" for i, chunk in enumerate(chunks)])
    prompt = (
        f"{system_prompt}{context}\n\n"
        f"User question:\n{question}\n\n"
        "Answer:"
    )
    # Model inference
    output = _model(
        prompt,
        max_tokens=512,
        temperature=0.2,
        top_p=0.95,
        stop=["</s>", "Question:", "User question:"],
        echo=False,
    )
    return output["choices"][0]["text"].strip()