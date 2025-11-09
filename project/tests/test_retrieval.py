from rag.embeddings import get_embedder
from rag.vectordb import build_chroma
from rag.retriever import build_retriever
from langchain_core.documents import Document
from pathlib import Path

def test_retriever_basic(tmp_path: Path):
    embedder = get_embedder("sentence-transformers/all-MiniLM-L6-v2")
    docs = [Document(page_content="The Mughal Empire was founded by Babur."),
            Document(page_content="Ashoka ruled the Maurya Empire.")]
    store = build_chroma(docs, embedder, tmp_path / "chroma")
    retriever = build_retriever(store, k=1)
    res = retriever.get_relevant_documents("Who founded the Mughal Empire?")
    assert len(res) >= 1
