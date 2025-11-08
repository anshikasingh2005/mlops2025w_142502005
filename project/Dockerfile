# ---- Base Image ----
FROM python:3.11-slim

# Set up working directory
WORKDIR /app

# Disable buffering for logs (for Spaces console output)
ENV PYTHONUNBUFFERED=1

# Install uv (fast dependency manager)
RUN pip install --no-cache-dir uv

# Copy dependency file and install
COPY uv.toml uv.lock ./
RUN uv sync --frozen --system .


# Copy the rest of the application code
COPY . .

# Expose Gradio’s default port
EXPOSE 7860

# Environment variables (these can be overridden by Spaces secrets)
ENV TGI_URL="meta-llama/Meta-Llama-3-8B-Instruct"
ENV HF_API_TOKEN=""
ENV CHROMA_DIR="./data/chroma"
ENV EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
ENV RETRIEVER_K="4"
# Default command: run Gradio app
CMD ["uv", "run", "python", "-m", "app.main"]
