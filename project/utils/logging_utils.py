from loguru import logger
import json, time, uuid
from pathlib import Path
import wandb

# --- Logging setup ---
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"
ERROR_FILE = LOG_DIR / "errors.jsonl"
INTERACTIONS_FILE = LOG_DIR / "interactions.jsonl"

# Configure Loguru
logger.add(LOG_FILE, rotation="10 MB", retention="7 days",
           format="{time} | {level} | {message}", level="INFO")

# --- WandB setup ---
wandb.init(project="students-companion", job_type="inference")

# --- Helper ---
SESSION_ID = str(uuid.uuid4())

def log_interaction_advanced(
    question: str,
    answer: str,
    latency: float,
    mode: str,
    retrieval_time: float,
    generation_time: float,
    sources: list,
    success: bool = True,
):
    """Logs interaction locally and sends metrics to W&B."""
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": SESSION_ID,
        "question": question,
        "answer_preview": answer[:200],
        "latency_s": round(latency, 3),
        "retrieval_time_s": round(retrieval_time, 3),
        "generation_time_s": round(generation_time, 3),
        "mode": mode,
        "num_sources": len(sources),
        "sources": sources,
        "success": success,
    }

    # Local JSONL storage
    with open(INTERACTIONS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

    # Structured log
    if success:
        logger.info(json.dumps(entry))
    else:
        logger.error(json.dumps(entry))

    # WandB live metrics
    wandb.log({
        "total_latency_s": latency,
        "retrieval_time_s": retrieval_time,
        "generation_time_s": generation_time,
        "num_sources": len(sources),
        "success": int(success),
        "mode": mode,
    })
    # --- trigger alerts based on thresholds ---
    maybe_trigger_alert("total_latency_s", latency, threshold=5.0, message=f"Slow response for mode={mode}")
    maybe_trigger_alert("error_rate", 1 - int(success), threshold=0.5, message="High failure rate detected.")


def maybe_trigger_alert(metric_name: str, value: float, threshold: float, message: str):
    """
    Trigger a W&B alert when a metric crosses a threshold.
    Works on all tiers, including Free.
    """
    try:
        if value > threshold:
            wandb.alert(
                title=f"{metric_name} Alert",
                text=f"{metric_name} = {value:.2f} exceeded threshold ({threshold})\nDetails: {message}",
                level=wandb.AlertLevel.WARN
            )
    except Exception as e:
        logger.warning(f"Failed to send W&B alert: {e}")





def log_error(question: str, e: Exception, trace: str = ""):
    """Logs errors separately."""
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": SESSION_ID,
        "question": question,
        "error": str(e),
        "trace": trace,
    }
    with open(ERROR_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    logger.error(json.dumps(record))
    wandb.log({"error": 1})
