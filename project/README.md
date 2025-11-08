# Student's Companion (6–12) – RAG Chatbot

A modular, MLOps-friendly RAG chatbot trained on NCERT (Class 6–12 History, English).  

- **GPU serving** with **Text Generation Inference (TGI)**
- **Chroma** vector store
- **Gradio** UI (ideal for Hugging Face Spaces)

## Run locally (dev)
```bash
uv sync
uv run python -m app.main
```

## Project structure
```
students-companion/
├─ app/
│  ├─ main.py
│  └─ ui.py
├─ rag/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ ingestion.py
│  ├─ splitting.py
│  ├─ embeddings.py
│  ├─ vectordb.py
│  ├─ retriever.py
│  ├─ prompts.py
│  ├─ generator.py
│  ├─ chain.py
│  └─ tasks.py
├─ data/
│  ├─ ncert/           
│  └─ chroma/          # persisted Chroma index
├─ eval/
│  ├─ datasets/
│  └─ evaluate_rag.py
├─ scripts/
│  ├─ build_index.py
│  └─ refresh_index.py
├─ tests/
│  ├─ test_ingestion.py
│  ├─ test_retrieval.py
│  └─ test_chain_smoke.py
├─ .github/workflows/ci.yml
├─ Dockerfile
├─ requirements.txt
├─ .env.example
└─ README.md
```S
