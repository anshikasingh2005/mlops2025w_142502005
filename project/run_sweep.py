import wandb
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from scipy.spatial.distance import cosine # <-- 1. NEW IMPORT

# --- Local Project Imports ---
from rag.config import settings
from rag.ingestion import load_pdfs
from rag.splitting import split_docs
from rag.embeddings import get_embedder
from rag.vectordb import build_chroma
from rag.retriever import build_retriever
from rag.generator import make_llm_tgi
from rag.tasks import run_task
# We DON'T need rag.chain.ask anymore
from evaluation import EVAL_QUESTIONS

load_dotenv()

# --- 1. Define Your Sweep Configuration ---
# (This is unchanged)
sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'average_score',
        'goal': 'maximize'
    },
    'parameters': {
        'chunk_size': {
            'values': [512, 1024]
        },
        'chunk_overlap': {
            'values': [50, 100]
        },
        'top_k': {
            'values': [3, 5]
        }
    }
}

# --- 2. Define the Semantic Similarity Scoring Function (NEW) ---
def score_answer_with_similarity(embedder, generated_answer, ground_truth):
    """
    Scores the answer by calculating the semantic similarity
    between the generated answer and the ground truth.
    """
    try:
        # 1. Embed both sentences
        g_truth_embedding = embedder.embed_query(ground_truth)
        gen_ans_embedding = embedder.embed_query(generated_answer)

        # 2. Calculate Cosine Similarity
        # scipy.distance.cosine returns the *distance* (0=identical, 1=different)
        # So, similarity = 1 - distance
        score = 1 - cosine(g_truth_embedding, gen_ans_embedding)
        
        return score
        
    except Exception as e:
        print(f"  [Scoring Error: {e}]")
        return 0 # Fail safe

# --- 3. Define the Main Experiment Function (UPDATED) ---
def evaluate_run():
    # Initialize a new W&B run
    run = wandb.init()
    
    config = run.config
    chunk_size = config.chunk_size
    chunk_overlap = config.chunk_overlap
    k = config.top_k

    print(f"--- Starting Run: chunk_size={chunk_size}, overlap={chunk_overlap}, k={k} ---")

    db_path = Path(f"./chroma_temp_{run.id}")

    try:
        # 1. SETUP: Build a NEW vector store for this run
        embedder = get_embedder(settings.EMBEDDING_MODEL) # We need this for scoring now
        llm = make_llm_tgi(settings.TGI_URL, settings.HF_API_TOKEN)
        
        docs = load_pdfs(settings.NCERT_DIR)
        chunks = split_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        store = build_chroma(chunks, embedder, db_path)
        retriever = build_retriever(store, k=k)

        # 2. EVALUATION: Test all questions
        columns = ["Question", "Ground Truth", "Generated Answer", "Score"]
        eval_table = wandb.Table(columns=columns)
        
        total_score = 0
        for item in EVAL_QUESTIONS:
            question = item["question"]
            ground_truth = item["ground_truth"]
            
            # This is the RAG call to get the answer
            generated_answer = run_task(llm, retriever, "NCERT", 10, question, None)
            
            # This is the NEW "Semantic Similarity" call to score the answer
            score = score_answer_with_similarity(embedder, generated_answer, ground_truth)
            total_score += score
            
            # Add data to our W&B Table
            eval_table.add_data(question, ground_truth, generated_answer, score)

        # 3. LOGGING: Log the final metrics for this run
        # The score will now be a float, e.g., 0.875
        average_score = total_score / len(EVAL_QUESTIONS)
        
        run.log({
            "average_score": average_score,
            "evaluation_results": eval_table
        })
        
        print(f"--- Finished Run. Average Score: {average_score:.4f} ---")

    except Exception as e:
        print(f"!!! RUN FAILED: {e} !!!")
        run.log({"average_score": 0})
    
    finally:
        # 4. CLEANUP: Clean up the temporary database
        try:
            if db_path.exists():
                shutil.rmtree(db_path)
                print(f"Cleaned up temporary database: {db_path}")
        except Exception as e:
            print(f"Warning: Could not clean up {db_path}. Error: {e}")

# --- 4. Define the Main Execution Block ---
# (This is unchanged)
if __name__ == "__main__":
    project_name = "NCERT-RAG-Sweep"  
    wandb.login()
    sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
    print(f"--- Starting W&B Sweep: {sweep_id} ---")
    wandb.agent(sweep_id, function=evaluate_run, count=8)
    print("--- Sweep finished! ---")