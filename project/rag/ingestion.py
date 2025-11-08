from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

def load_pdfs(dir_path: Path):
    docs = []
    if not dir_path.exists():
        return docs
    for pdf in sorted(dir_path.glob("*.pdf")):
        docs.extend(PyPDFLoader(str(pdf)).load())
    return docs

def load_uploads(filepaths: list[str]):
    docs = []
    for p in filepaths:
        docs.extend(PyPDFLoader(p).load())
    return docs
