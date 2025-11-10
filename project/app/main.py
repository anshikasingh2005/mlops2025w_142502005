import os
import gradio as gr
from dotenv import load_dotenv
load_dotenv()

from rag.config import settings
from rag.ingestion import load_pdfs, load_uploads
from rag.splitting import split_docs
from rag.embeddings import get_embedder
from rag.vectordb import load_chroma, build_chroma, upsert_chunks
from rag.retriever import build_retriever
from rag.generator import make_llm_tgi
from rag.tasks import run_task
from app.ui import build_ui

import time, traceback
from utils.logging_utils import log_interaction_advanced, log_error
from rag.tasks import run_task



print("HF_API_TOKEN loaded:", os.getenv("HF_API_TOKEN"))
import os
print("🔐 W&B key loaded:", bool(os.getenv("WANDB_API_KEY")))


# Bootstrapping index
embedder = get_embedder(settings.EMBEDDING_MODEL)
try:
    store = load_chroma(embedder, settings.CHROMA_DIR)
    # If empty, attempt initial build from NCERT (if present)
    if store._collection.count() == 0 and settings.NCERT_DIR.exists():
        docs = load_pdfs(settings.NCERT_DIR)
        chunks = split_docs(docs)
        store = build_chroma(chunks, embedder, settings.CHROMA_DIR)
except Exception:
    # First run
    docs = load_pdfs(settings.NCERT_DIR)
    chunks = split_docs(docs) if docs else []
    store = build_chroma(chunks, embedder, settings.CHROMA_DIR)

retriever = build_retriever(store, k=settings.RETRIEVER_K)

# --- This is the new, fast-loading code ---
'''
# 1. Load the embedder
embedder = get_embedder(settings.EMBEDDING_MODEL)

# 2. Load the pre-built database
print("Loading persistent ChromaDB...")
store = load_chroma(embedder, settings.CHROMA_DIR)
print(f"ChromaDB loaded with {store._collection.count()} documents.")

# 3. Build the retriever
retriever = build_retriever(store, k=settings.RETRIEVER_K)

# --- End of new code ---
'''
# LLM via TGI
llm = make_llm_tgi(settings.TGI_URL, settings.HF_API_TOKEN)

#demo, mode, grade, files, chat, txt = build_ui()

'''
#respond function before logging code
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
'''


def respond(message, history, mode, grade, files):
    total_start = time.time()
    retrieval_time = generation_time = 0
    sources = []
    try:
        # Uploads if any
        if files and len(files) > 0:
            paths = [f.name for f in files]
            docs = load_uploads(paths)
            chunks = split_docs(docs)
            upsert_chunks(store, chunks)

        # --- Measure retrieval + generation ---
        retrieval_start = time.time()
        answer = run_task(llm, retriever, mode, int(grade), message, history)
        retrieval_time = time.time() - retrieval_start

        # (LLM generation timing if separate call)
        generation_start = time.time()
        # Suppose generation happens inside run_task → reuse retrieval_time
        generation_time = time.time() - generation_start

        total_latency = time.time() - total_start
        log_interaction_advanced(
            question=message,
            answer=answer,
            latency=total_latency,
            mode=mode,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            sources=[],  # update if run_task returns them
        )
        # --- Sanitize answer for Gradio display ---
        if isinstance(answer, (tuple, list)):
            # Usually (text, sources)
            answer_text = str(answer[0]) if len(answer) > 0 else ""
        else:
            answer_text = str(answer)
        history = (history or []) + [[message, answer_text]]
        return history, ""
        '''
        # --- ✅ FIX: ensure answer is always a string ---
        if isinstance(answer, list):
            answer_text = "\n\n".join([a for a in answer if isinstance(a, str)])
        else:
            answer_text = str(answer)

        history = (history or []) + [[message, answer]]
        return history, ""
        '''
    except Exception as e:
        trace = traceback.format_exc()
        total_latency = time.time() - total_start
        log_error(message, e, trace)
        log_interaction_advanced(
            message, str(e), total_latency, mode,
            retrieval_time, generation_time, [],
            success=False,
        )
        return history, "⚠️ Sorry, something went wrong."






#txt.submit(respond, [txt, chat, mode, grade, files], [chat, txt])
demo = build_ui(respond)

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=int(os.getenv("PORT", 7860)))
