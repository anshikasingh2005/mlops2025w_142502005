'''
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
'''

from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from huggingface_hub import snapshot_download
from rag.config import settings
import os

# --- This is the new, "smart" loading function ---
def load_chroma(embedder: Embeddings, persist_dir: Path) -> Chroma:
    """
    Loads the Chroma database. If it doesn't exist locally,
    it downloads it from the public Hugging Face Hub.
    """
    # Check if the database already exists locally
    if persist_dir.exists() and any(persist_dir.iterdir()):
        print(f"Loading existing local database from '{persist_dir}'")
    else:
        # If it doesn't exist, download it from the Hub
        print(f"--- Local DB not found at '{persist_dir}'.")
        print(f"Downloading database from Hugging Face Hub: {settings.DATASET_REPO_ID}...")

        try:
            # Download the entire dataset repository to the specified directory
            # We set token=None because it is a public repository
            snapshot_download(
                repo_id=settings.DATASET_REPO_ID,
                repo_type="dataset",
                local_dir=persist_dir,
                token=None  # <-- This is the change for public repos
            )
            print(f"Download complete. Database is at '{persist_dir}'")
        except Exception as e:
            print(f"Failed to download database: {e}")
            raise

    # Load the database from the (now existing) directory
    return Chroma(embedding_function=embedder, persist_directory=str(persist_dir))


def build_chroma(chunks: list[Document], embedder: Embeddings, persist_dir: Path) -> Chroma:
    """Builds a new Chroma database from a list of documents."""
    # Ensure the directory exists
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the DB from documents
    return Chroma.from_documents(chunks, embedding=embedder, persist_directory=str(persist_dir))


def upsert_chunks(store: Chroma, chunks: list[Document]) -> Chroma:
    """
    Adds new documents (chunks) to an existing Chroma store.
    The .persist() call is no longer needed in new versions.
    """
    store.add_documents(chunks)
    # store.persist()  <--- THIS LINE WAS REMOVED. This is the fix.
    return store