import os
import gradio as gr
from dotenv import load_dotenv

from rag.config import settings
from rag.ingestion import load_pdfs, load_uploads
from rag.splitting import split_docs
from rag.embeddings import get_embedder
from rag.vectordb import load_chroma, build_chroma, upsert_chunks
from rag.retriever import build_retriever
from rag.generator import make_llm_tgi
from rag.tasks import run_task
from app.ui import build_ui

load_dotenv()

print("HF_API_TOKEN loaded:", os.getenv("HF_API_TOKEN"))


# Bootstrapping index
# embedder = get_embedder(settings.EMBEDDING_MODEL)
# try:
#     store = load_chroma(embedder, settings.CHROMA_DIR)
#     # If empty, attempt initial build from NCERT (if present)
#     if store._collection.count() == 0 and settings.NCERT_DIR.exists():
#         docs = load_pdfs(settings.NCERT_DIR)
#         chunks = split_docs(docs)
#         store = build_chroma(chunks, embedder, settings.CHROMA_DIR)
# except Exception:
#     # First run
#     docs = load_pdfs(settings.NCERT_DIR)
#     chunks = split_docs(docs) if docs else []
#     store = build_chroma(chunks, embedder, settings.CHROMA_DIR)

# retriever = build_retriever(store, k=settings.RETRIEVER_K)

# --- This is the new, fast-loading code ---

# 1. Load the embedder
embedder = get_embedder(settings.EMBEDDING_MODEL)

# 2. Load the pre-built database
print("Loading persistent ChromaDB...")
store = load_chroma(embedder, settings.CHROMA_DIR)
print(f"✅ ChromaDB loaded with {store._collection.count()} documents.")

# 3. Build the retriever
retriever = build_retriever(store, k=settings.RETRIEVER_K)

# --- End of new code ---

# LLM via TGI
llm = make_llm_tgi(settings.TGI_URL, settings.HF_API_TOKEN)

#demo, mode, grade, files, chat, txt = build_ui()

def respond(message, history, mode, grade, files):
    # Ingest uploads if any
    if files and len(files) > 0:
        paths = [f.name for f in files]
        docs = load_uploads(paths)
        chunks = split_docs(docs)
        upsert_chunks(store, chunks)
    answer = run_task(llm, retriever, mode, int(grade), message, history)
    history = (history or []) + [[message, answer]]
    return history, ""

#txt.submit(respond, [txt, chat, mode, grade, files], [chat, txt])
demo = build_ui(respond)

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=int(os.getenv("PORT", 7860)))
