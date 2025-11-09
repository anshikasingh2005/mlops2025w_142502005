from pathlib import Path
from rag.ingestion import load_pdfs

def test_load_pdfs_empty(tmp_path: Path):
    docs = load_pdfs(tmp_path)
    assert docs == []
