# 🦙 Research Assistant LLM (arXiv + Document Q&A)

A simple but powerful Streamlit app that lets you:
- 🔎 Search arXiv for research papers by topic.
- 📄 Upload your own PDF/TXT research papers.
- 💬 Ask questions about your documents using retrieval-augmented generation (RAG) powered by a fully local LLM (DeepSeek-Coder-1.3B-Instruct, fallback to TinyLlama-1.1B-Chat).

## 🚀 Quick Start (Streamlit Cloud)

1. **Fork or clone this repo** to your GitHub.
2. **[Deploy to Streamlit Cloud](https://streamlit.io/cloud)**:
   - New app → point to `app/app.py`
   - No secrets required (everything runs locally).
3. **First launch will auto-download the LLM model** (~800MB). If memory-constrained, app falls back to TinyLlama (~700MB).
4. **Enjoy!**
   - Use the sidebar to search arXiv.
   - Upload PDFs or TXTs (max 10MB each).
   - Ask questions in the chat box.

## 🛠️ Local Development

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run app/app.py
```

*On first run, models auto-download under `/models/`.*

## 💡 Features

- **Topic Search:** Find and preview papers from arXiv.
- **Document Upload:** Extracts and chunks your PDFs for efficient semantic search.
- **In-memory FAISS index:** Fast, private, session-based retrieval.
- **Local LLM QA:** No OpenAI/GPT API fees. All Q&A runs on your CPU.

## 📝 Limitations

- Free Streamlit Cloud has ~1GB RAM & no GPU. Large models or big batch uploads may hit memory limits.
- Uploaded files are not persisted between sessions. Uploaded content is never sent to a third-party server.
- For best results, upload academic-style PDFs or clean text.

## 📦 Repo structure

```
app/
    app.py              # Streamlit UI
backend/
    arxiv_search.py     # arXiv API queries
    pdf_extract.py      # PDF/TXT extraction
    chunker.py          # Text chunking
    embedder.py         # Sentence-transformers embeddings
    vector_db.py        # FAISS helpers
    rag_engine.py       # Retrieval & Llama answer generation
models/                 # Models auto-download here
requirements.txt
README.md
```

## 🤝 Credits

- [DeepSeek Coder](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct-GGUF)
- [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0-GGUF)
- [arxiv](https://github.com/lukasschwab/arxiv.py)
- [pymupdf](https://github.com/pymupdf/PyMuPDF)
- [sentence-transformers](https://www.sbert.net/)
- [faiss](https://github.com/facebookresearch/faiss)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

---

MIT License