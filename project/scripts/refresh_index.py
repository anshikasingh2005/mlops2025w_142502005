from rag.config import settings
from rag.ingestion import load_uploads
from rag.splitting import split_docs
from rag.embeddings import get_embedder
from rag.vectordb import load_chroma, upsert_chunks

def main(paths: list[str]):
    embedder = get_embedder(settings.EMBEDDING_MODEL)
    store = load_chroma(embedder, settings.CHROMA_DIR)
    docs = load_uploads(paths)
    chunks = split_docs(docs)
    upsert_chunks(store, chunks)
    print("Upserted documents into Chroma index.")

if __name__ == "__main__":
    # Example usage:
    # python scripts/refresh_index.py path1.pdf path2.pdf
    import sys
    main(sys.argv[1:])
