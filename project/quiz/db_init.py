import os
from pathlib import Path
from chromadb import PersistentClient

def get_data():
    # Always build path relative to this file location
    base_dir = Path(__file__).resolve().parent.parent  # go up from /quiz to /project
    chroma_db_path = base_dir / "data" / "chroma"

    chroma_db_path = chroma_db_path.resolve()

    if not chroma_db_path.exists():
        raise FileNotFoundError(f"❌ ChromaDB folder not found at: {chroma_db_path}")

    print(f"✅ Connected to ChromaDB at:\n{chroma_db_path}")
    
    client = PersistentClient(path=str(chroma_db_path))
    collection = client.get_collection(name="langchain")


    return collection

client = get_data() 
print(client)
