# dashboard/model_utils.py
import os
import torch
from huggingface_hub import InferenceApi
from typing import Dict, Any

# Local model loader: expects a local folder containing config and model weights
def load_local_model(local_dir: str):
    """
    Loads a local PyTorch transformer model and tokenizer.
    Returns: (model, tokenizer, device)
    """
    from transformers import AutoTokenizer, AutoModel
    from finbert_model import MultiTaskFinBERT

    if not os.path.isdir(local_dir):
        raise FileNotFoundError(f"Local model dir not found: {local_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_dir, use_fast=False)
    # encoder
    encoder = AutoModel.from_pretrained(local_dir)
    model = MultiTaskFinBERT(encoder)
    # try to load weights (pytorch_model.bin or model.safetensors)
    # prefer safetensors if present
    from huggingface_hub import hf_hub_download
    # if user gave a path, transformers AutoModel.from_pretrained already attempts to load weights.
    model.to(device)
    model.eval()
    return model, tokenizer, device

# Local prediction wrapper
def make_local_prediction(text: str, model, tokenizer, device: str) -> Dict[str, Any]:
    """
    Run inference on local model. Return a dictionary with keys like 'sentiment' and 'tone'.
    This function assumes the MultiTaskFinBERT exposes a forward that returns predictions.
    Adapt to your MultiTaskFinBERT implementation.
    """
    import torch
    # Basic pipeline: tokenize, run model, post-process.
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        output = model.predict(inputs) if hasattr(model, "predict") else model(**inputs)
    # Postprocessing — this depends on your model output format.
    # Here we provide a generic wrapper expecting model to return dict or tuple.
    if isinstance(output, dict):
        return output
    # Example fallback: treat output as logits tuple (sentiment_logits, tone_logits)
    if isinstance(output, (list, tuple)) and len(output) >= 2:
        s_logits, t_logits = output[0], output[1]
        s_label = int(torch.argmax(s_logits, dim=-1).item()) if hasattr(s_logits, "argmax") else None
        t_label = int(torch.argmax(t_logits, dim=-1).item()) if hasattr(t_logits, "argmax") else None
        return {"sentiment_label": s_label, "tone_label": t_label}
    return {"raw_output": str(output)}

# HF Inference API wrapper
def make_hf_api_prediction(text: str, repo_id: str, token: str = None) -> Dict[str, Any]:
    """
    Use HuggingFace Inference API to query the remote model.
    This works for text-classification tasks or custom model endpoints.
    If model returns several labels or a dict, the function will try to return sensible fields.
    """
    try:
        client = InferenceApi(repo_id, token=token) if token else InferenceApi(repo_id)
        # call model
        resp = client(inputs=text)
        # Typical responses:
        # - list of dicts: [{"label":"POS","score":0.9}, ...]
        # - dict with multiple keys if your model returns a dict
        if isinstance(resp, list):
            # convert to simple mapping: label -> score (take highest)
            top = max(resp, key=lambda r: r.get("score", 0))
            return {"label": top.get("label"), "score": top.get("score")}
        elif isinstance(resp, dict):
            return resp
        else:
            return {"raw": resp}
    except Exception as e:
        return {"error": str(e)}

# optional helper to list or validate repo (not required)
def hf_repo_list(repo_id: str):
    from huggingface_hub import hf_api
    api = hf_api.HfApi()
    return api.model_info(repo_id)
