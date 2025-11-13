import os
from pathlib import Path
from chromadb import PersistentClient
from huggingface_hub import snapshot_download

def get_data():
    """
    Load ChromaDB collection either from local folder (if present)
    or directly from Hugging Face Hub dataset (for Spaces deployment).
    """
    base_dir = Path(__file__).resolve().parent.parent  # go up from /quiz to /project
    local_chroma_path = base_dir / "data" / "chroma"
    local_chroma_path = local_chroma_path.resolve()

    # Hugging Face dataset repo ID
    hf_dataset_id = "Shivani4444/mlops-ragsystem-chroma"

    # 1️⃣ Prefer local path if it exists
    if local_chroma_path.exists():
        print(f"✅ Connected to local ChromaDB at:\n{local_chroma_path}")
        client = PersistentClient(path=str(local_chroma_path))
        collection = client.get_collection(name="langchain")
        return collection

    # 2️⃣ Otherwise, download dataset snapshot from Hugging Face Hub
    print(f"🌐 Local ChromaDB not found. Downloading from Hugging Face Hub ({hf_dataset_id})...")
    try:
        hf_cache_dir = snapshot_download(
            repo_id=hf_dataset_id,
            repo_type="dataset",
            local_dir="/tmp/chroma_db_local",  # temporary storage for Spaces
            local_dir_use_symlinks=False,
        )
        print(f"✅ Downloaded ChromaDB dataset to: {hf_cache_dir}")
        client = PersistentClient(path=hf_cache_dir)
        collection = client.get_collection(name="langchain")
        return collection
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load ChromaDB from Hugging Face: {e}")
