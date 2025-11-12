from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Paths
    CHROMA_DIR: Path = Path("data/chroma")
    NCERT_DIR: Path = Path("data/ncert")

    # Embeddings & retrieval
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RETRIEVER_K: int = 4

    # Generation via TGI
    TGI_URL: str = "http://localhost:8080"
    HF_API_TOKEN: str | None = None
    WANDB_API_KEY: str | None = None  
    DATASET_REPO_ID: str = "Shivani4444/mlops-ragsystem-chroma"
    class Config:
        env_file = ".env"

settings = Settings()
 

# settings = Settings()

import json

# This is our 'promoted' config file from the W&B sweep
BEST_CONFIG_FILE = Path("best_config.json")

# Check if the file exists in the root directory
if BEST_CONFIG_FILE.exists() and BEST_CONFIG_FILE.is_file():
    print(f"---")
    print(f"--- Loading 'best_config.json' from W&B sweep ---")
    try:
        with open(BEST_CONFIG_FILE, "r") as f:
            best_config = json.load(f)
        
        # --- Overwrite settings object with new values ---
        
        # 1. Map 'top_k' from the sweep to 'RETRIEVER_K' in the app
        #    We use .get() to provide a fallback to the original value
        settings.RETRIEVER_K = best_config.get('top_k', settings.RETRIEVER_K)
        
        print(f"--- Overwriting RETRIEVER_K with: {settings.RETRIEVER_K} ---")

        # Note: chunk_size and chunk_overlap are "build-time" parameters.
        # Your app.py loads a pre-built DB, so it doesn't use them.
        # This is correct.
        print(f"---")

    except Exception as e:
        print(f"Warning: Could not load best_config.json. Using defaults. Error: {e}")
else:
    print("--- No 'best_config.json' found. Using default settings. ---")
