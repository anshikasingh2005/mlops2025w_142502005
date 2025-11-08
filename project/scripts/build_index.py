from rag.config import settings
from rag.ingestion import load_pdfs
from rag.splitting import split_docs
from rag.embeddings import get_embedder
from rag.vectordb import build_chroma

def main():
    docs = load_pdfs(settings.NCERT_DIR)
    if not docs:
        print("No PDFs found in", settings.NCERT_DIR)
        return
    chunks = split_docs(docs)
    embedder = get_embedder(settings.EMBEDDING_MODEL)
    store = build_chroma(chunks, embedder, settings.CHROMA_DIR)
    print("Chroma index built at:", settings.CHROMA_DIR)

if __name__ == "__main__":
    main()
