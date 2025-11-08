from pathlib import Path
from langchain_community.vectorstores import Chroma

def build_chroma(chunks, embedder, persist_dir: Path):
    persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(chunks, embedding=embedder, persist_directory=str(persist_dir))

def load_chroma(embedder, persist_dir: Path):
    persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(embedding_function=embedder, persist_directory=str(persist_dir))

def upsert_chunks(store: Chroma, chunks):
    store.add_documents(chunks)
    store.persist()
    return store
