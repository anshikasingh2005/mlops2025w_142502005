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

    class Config:
        env_file = ".env"

settings = Settings()
