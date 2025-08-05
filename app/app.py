import os
import streamlit as st
from typing import List
import tempfile

from backend.arxiv_search import search_arxiv
from backend.pdf_extract import extract_text
from backend.chunker import chunk_text
from backend.embedder import embed_texts
from backend.vector_db import create_index, search, add_embeddings
from backend.rag_engine import answer_question, get_model_id, ensure_model_loaded

st.set_page_config(page_title="Research Assistant", layout="wide")
st.title("📚 Research Assistant: arXiv + Document Q&A (RAG)")

# Session state
if "docs_uploaded" not in st.session_state:
    st.session_state.docs_uploaded = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "model_id" not in st.session_state:
    st.session_state.model_id = None

# -------- Sidebar: arXiv Search --------
with st.sidebar:
    st.header("🔎 arXiv Topic Search")
    query = st.text_input("Enter a research topic", "")
    max_results = st.slider("Max results", 1, 25, 10)
    if st.button("Search arXiv") and query.strip():
        with st.spinner("Searching arXiv..."):
            results = search_arxiv(query, max_results)
        if results:
            st.success(f"Found {len(results)} papers.")
            for i, paper in enumerate(results, 1):
                st.markdown(f"**{i}. [{paper['title']}]({paper['pdf_url']})**  \n"
                            f"*Authors*: {', '.join(paper['authors'])}  \n"
                            f"*Published*: {paper['published']}  \n"
                            f"{paper['abstract'][:400]}{'...' if len(paper['abstract']) > 400 else ''}")
                st.markdown("---")
        else:
            st.warning("No results found.")

# -------- Main: File Upload & Index --------
st.header("📄 Upload Research Papers (PDF/TXT)")
uploaded_files = st.file_uploader(
    "Upload one or more PDF or TXT files (max 10 MB each)", 
    type=["pdf", "txt"], 
    accept_multiple_files=True
)

if uploaded_files:
    # Only process new uploads
    new_files = [f for f in uploaded_files if f.name not in st.session_state.docs_uploaded]
    if new_files:
        texts = []
        for file in new_files:
            if file.size > 10 * 1024 * 1024:
                st.error(f"{file.name} is too large (max 10 MB).")
                continue
            st.info(f"Extracting text from {file.name}...")
            # Save to temp file for PyMuPDF
            if file.type == "application/pdf":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.read())
                    tmp.flush()
                    text = extract_text(tmp.name)
                os.unlink(tmp.name)
            else:
                text = file.read().decode("utf-8", errors="replace")
            if not text.strip():
                st.error(f"Failed to extract text from {file.name} or document is empty.")
                continue
            texts.append(text)
            st.session_state.docs_uploaded.append(file.name)
        # Chunk and embed
        all_chunks = []
        for text in texts:
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
        if all_chunks:
            with st.spinner("Embedding and indexing..."):
                chunk_embs = embed_texts(all_chunks)
                if st.session_state.embeddings is None:
                    st.session_state.embeddings = chunk_embs
                    st.session_state.chunks = all_chunks
                    st.session_state.faiss_index = create_index(chunk_embs)
                else:
                    st.session_state.embeddings = np.vstack([st.session_state.embeddings, chunk_embs])
                    st.session_state.chunks.extend(all_chunks)
                    add_embeddings(st.session_state.faiss_index, chunk_embs)
            st.success(f"Processed {len(all_chunks)} new chunks.")
else:
    st.info("No files uploaded yet. Upload research papers above to enable Q&A.")

# -------- Chat QA --------
import numpy as np

st.markdown("### 💬 Ask a question about your uploaded documents")
user_question = st.text_input("Ask a question and press Enter", key="question")

if user_question and st.session_state.faiss_index is not None:
    # Ensure model is loaded (loads only once per session)
    with st.spinner("Loading language model..."):
        model_id = ensure_model_loaded()
        st.session_state.model_id = model_id
    # Embed question, retrieve top-k
    q_emb = embed_texts([user_question])[0]
    idxs = search(st.session_state.faiss_index, q_emb, top_k=5)
    retrieved = [st.session_state.chunks[i] for i in idxs]
    # Answer using LLM
    with st.spinner("Generating answer..."):
        answer = answer_question(user_question, retrieved)
    st.markdown(f"**Answer:** {answer}")
    with st.expander("Show retrieved source text"):
        for i, chunk in enumerate(retrieved, 1):
            st.markdown(f"**Chunk {i}:**\n\n{chunk}\n")
elif user_question:
    st.warning("Please upload and process at least one document to enable Q&A.")

# -------- Model Info --------
if st.session_state.model_id:
    st.caption(f"LLM in use: `{st.session_state.model_id}`")

st.markdown("---")
st.markdown(
    "Tips: Use the arXiv search in the sidebar to find papers. "
    "Upload your own PDFs or TXTs to ask detailed questions using Retrieval-Augmented Generation (RAG)."
)