from typing import Optional
from langchain_huggingface import HuggingFaceEndpoint
'''
def make_llm_tgi(tgi_url: str, hf_api_token: Optional[str] = None):
    # HuggingFaceTextGenInference expects an endpoint URL to a TGI server
    # If auth is required, set HUGGINGFACEHUB_API_TOKEN env var OR pass in headers via client kwargs.
    headers = None
    if hf_api_token:
        headers = {"Authorization": f"Bearer {hf_api_token}"}
    return HuggingFaceTextGenInference(
        inference_server_url=tgi_url,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.9,
        repetition_penalty=1.05,
        client=None,
        # pass headers if needed
        # model_kwargs={"headers": headers}  # (depends on your deployment)
    )

def make_llm_tgi(endpoint_url, hf_token):
    """Connect to a Hugging Face Text Generation Inference (TGI) endpoint."""
    return HuggingFaceEndpoint(
        endpoint_url=endpoint_url,
        huggingfacehub_api_token=hf_token,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.7,
    )
# rag/generator.py

# rag/generator.py
from langchain_huggingface import HuggingFaceEndpoint


def make_llm_tgi(endpoint_url, hf_token):
    """
    Uses Hugging Face hosted inference API for text generation.
    Works with models like Mistral, Zephyr, Gemma, etc.
    """

    print(f"Using Hugging Face hosted model: {endpoint_url}")

    return HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.7,
        huggingfacehub_api_token=hf_token,
    )
'''
# rag/generator.py
from huggingface_hub import InferenceClient
from langchain_core.language_models import LLM
from pydantic import PrivateAttr
from typing import Optional, List


class HuggingFaceChatLLM(LLM):
    """Custom LLM wrapper for Hugging Face chat models (e.g., Llama 3)."""

    _client: InferenceClient = PrivateAttr()

    def __init__(self, model_name: str, hf_token: str):
        super().__init__()
        self._client = InferenceClient(model_name, token=hf_token)
        print(f"✅ Using Hugging Face chat model: {model_name}")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Send the prompt using the HF chat_completion API."""
        try:
            response = self._client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.7,
            )
            return response.choices[0].message["content"]
        except Exception as e:
            return f"⚠️ Error communicating with model: {e}"

    @property
    def _identifying_params(self):
        return {"model": "meta-llama/Meta-Llama-3-8B-Instruct"}

    @property
    def _llm_type(self):
        return "huggingface_chat"


def make_llm_tgi(model_name: str, hf_token: str):
    """Factory that creates a conversational Llama 3 LLM."""
    if not hf_token:
        raise ValueError("Missing HF_API_TOKEN in .env file!")
    return HuggingFaceChatLLM(model_name, hf_token)
