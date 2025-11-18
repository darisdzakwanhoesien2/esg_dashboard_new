# dashboard/model_utils.py
import os
import json
from typing import Any, Dict, List, Tuple, Optional

# light dependency used in both envs
from huggingface_hub import InferenceApi
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", None)
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "darisdzakwanhoesien/bertesg_tone")


# -------------------------
# Environment detection
# -------------------------
def torch_available() -> bool:
    """
    Return True if torch is importable in this environment.
    We import inside the function to avoid top-level import errors on Streamlit Cloud.
    """
    try:
        import importlib

        spec = importlib.util.find_spec("torch")
        return spec is not None
    except Exception:
        return False


# -------------------------
# Hugging Face Inference API (cloud-safe)
# -------------------------
def get_hf_inference_client(repo_id: Optional[str] = None, token: Optional[str] = None) -> InferenceApi:
    """
    Return an InferenceApi client configured with token (if provided)
    """
    repo = repo_id or HF_MODEL_REPO
    tok = token or HF_TOKEN
    client = InferenceApi(repo, token=tok)
    return client


def make_hf_api_prediction(texts: List[str], repo_id: Optional[str] = None, token: Optional[str] = None, timeout: int = 120) -> Any:
    """
    Make a request to the HF Inference API for the given texts.
    Returns the raw response (model-specific).
    """
    client = get_hf_inference_client(repo_id=repo_id, token=token)
    # Some inference endpoints accept list inputs; do best-effort.
    if len(texts) == 1:
        payload = texts[0]
    else:
        payload = texts

    # call the inference api (this will vary depending on your repo/model)
    res = client(inputs=payload, params={"use_cache": False}, timeout=timeout)
    return res


# -------------------------
# Local model loading & prediction (DEFER imports)
# -------------------------
def load_local_model_from_hub(repo_id: str, local_weights_path: Optional[str] = None, device: Optional[str] = None):
    """
    Load a model either by downloading HF weights or using local path.
    This function intentionally imports torch/transformers only when called (local dev).
    Returns model, tokenizer, device.
    """
    # local-heavy imports
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, AutoModel

    # device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # tokenizer + encoder
    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=False)
    encoder = AutoModel.from_pretrained(repo_id)

    # wrap into your MultiTaskFinBERT if you have it locally.
    # The app expects a model with a .predict(texts) or similar.
    # User should adapt to their MultiTaskFinBERT implementation.
    try:
        # if your project defines MultiTaskFinBERT in a local module, import it here
        from model_impl import MultiTaskFinBERT  # optional local implementation name
    except Exception:
        MultiTaskFinBERT = None

    model = None
    if MultiTaskFinBERT is not None:
        model = MultiTaskFinBERT(encoder)
    else:
        # fallback to encoder (user will likely adapt)
        model = encoder

    # if user provided a local weights path (prefer this), else try hub download
    if local_weights_path:
        state = torch.load(local_weights_path, map_location=device)
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            # if state is a dict with "model_state_dict" style, attempt common keys
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        # attempt to download typical HF weight filename
        try:
            model_file = hf_hub_download(repo_id, filename="pytorch_model.bin")
            state = torch.load(model_file, map_location=device)
            model.load_state_dict(state, strict=False)
        except Exception:
            # best-effort: model may already have its own HF weights loaded via from_pretrained
            pass

    model.to(device)
    model.eval()

    return model, tokenizer, device


def make_local_prediction(model, tokenizer, texts: List[str], device: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Make predictions locally using a model/tokenizer. This is intentionally generic:
    - If your MultiTaskFinBERT has a forward/predict method, you can adapt this function.
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Example: if model is a transformers encoder, just return tokenized inputs as placeholder
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**enc)  # may vary by model type

    # Convert outputs to a serializable placeholder response — user should customize.
    return [{"raw_output": None, "note": "Customize make_local_prediction to return model outputs"}]
