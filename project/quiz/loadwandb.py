import wandb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

def load_wandb_model(
    project_name: str,
    artifact_path: str,
    save_dir: str = "mlops2025w_142502005/project/quiz",
    job_name: str = "inference"
):
    """
    Downloads a fine-tuned model from W&B into a specific project directory
    and loads it for inference.

    Args:
        project_name (str): W&B project name.
        artifact_path (str): "<entity>/<project>/<artifact>:<version>"
        save_dir (str): Local directory where model should be downloaded.
        job_name (str): W&B run job type.

    Returns:
        tokenizer, model, downloaded_model_path
    """

    # Expand and create directory if needed
    save_dir = os.path.abspath(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)

    wandb.login()
    run = wandb.init(project=project_name, job_type=job_name)

    print(f"⬇️ Downloading model artifact to: {save_dir}")

    try:
        artifact = run.use_artifact(artifact_path, type='model')

        # ✅ Download artifact specifically into your target folder
        model_dir = artifact.download(root=save_dir)

        print(f"✅ Model downloaded to: {model_dir}")
    except Exception as e:
        print(f"❌ Failed to download artifact:\n{e}")
        return None, None, None

    # Load tokenizer and model from the downloaded directory
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    print("✅ Model and tokenizer successfully loaded.")

    return tokenizer, model, model_dir

project_name = "bart_qna_generator"
artifact_path = "142502009_/bart_qna_generator/model-2en2diu6:v5" 
load_wandb_model(project_name, artifact_path)