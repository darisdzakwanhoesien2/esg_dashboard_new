# # dashboard/model_utils.py
# import os
# import json
# from typing import Any, Dict, List, Tuple, Optional

# # light dependency used in both envs
# from huggingface_hub import InferenceApi
# from dotenv import load_dotenv

# load_dotenv()

# HF_TOKEN = os.getenv("HF_TOKEN", None)
# HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "darisdzakwanhoesien/bertesg_tone")


# # -------------------------
# # Environment detection
# # -------------------------
# def torch_available() -> bool:
#     """
#     Return True if torch is importable in this environment.
#     We import inside the function to avoid top-level import errors on Streamlit Cloud.
#     """
#     try:
#         import importlib

#         spec = importlib.util.find_spec("torch")
#         return spec is not None
#     except Exception:
#         return False


# # -------------------------
# # Hugging Face Inference API (cloud-safe)
# # -------------------------
# def get_hf_inference_client(repo_id: Optional[str] = None, token: Optional[str] = None) -> InferenceApi:
#     """
#     Return an InferenceApi client configured with token (if provided)
#     """
#     repo = repo_id or HF_MODEL_REPO
#     tok = token or HF_TOKEN
#     client = InferenceApi(repo, token=tok)
#     return client


# def make_hf_api_prediction(texts: List[str], repo_id: Optional[str] = None, token: Optional[str] = None, timeout: int = 120) -> Any:
#     """
#     Make a request to the HF Inference API for the given texts.
#     Returns the raw response (model-specific).
#     """
#     client = get_hf_inference_client(repo_id=repo_id, token=token)
#     # Some inference endpoints accept list inputs; do best-effort.
#     if len(texts) == 1:
#         payload = texts[0]
#     else:
#         payload = texts

#     # call the inference api (this will vary depending on your repo/model)
#     res = client(inputs=payload, params={"use_cache": False}, timeout=timeout)
#     return res


# # -------------------------
# # Local model loading & prediction (DEFER imports)
# # -------------------------
# def load_local_model_from_hub(repo_id: str, local_weights_path: Optional[str] = None, device: Optional[str] = None):
#     """
#     Load a model either by downloading HF weights or using local path.
#     This function intentionally imports torch/transformers only when called (local dev).
#     Returns model, tokenizer, device.
#     """
#     # local-heavy imports
#     import torch
#     from huggingface_hub import hf_hub_download
#     from transformers import AutoTokenizer, AutoModel

#     # device
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"

#     # tokenizer + encoder
#     tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=False)
#     encoder = AutoModel.from_pretrained(repo_id)

#     # wrap into your MultiTaskFinBERT if you have it locally.
#     # The app expects a model with a .predict(texts) or similar.
#     # User should adapt to their MultiTaskFinBERT implementation.
#     try:
#         # if your project defines MultiTaskFinBERT in a local module, import it here
#         from model_impl import MultiTaskFinBERT  # optional local implementation name
#     except Exception:
#         MultiTaskFinBERT = None

#     model = None
#     if MultiTaskFinBERT is not None:
#         model = MultiTaskFinBERT(encoder)
#     else:
#         # fallback to encoder (user will likely adapt)
#         model = encoder

#     # if user provided a local weights path (prefer this), else try hub download
#     if local_weights_path:
#         state = torch.load(local_weights_path, map_location=device)
#         try:
#             model.load_state_dict(state, strict=False)
#         except Exception:
#             # if state is a dict with "model_state_dict" style, attempt common keys
#             if isinstance(state, dict) and "model_state_dict" in state:
#                 model.load_state_dict(state["model_state_dict"], strict=False)
#     else:
#         # attempt to download typical HF weight filename
#         try:
#             model_file = hf_hub_download(repo_id, filename="pytorch_model.bin")
#             state = torch.load(model_file, map_location=device)
#             model.load_state_dict(state, strict=False)
#         except Exception:
#             # best-effort: model may already have its own HF weights loaded via from_pretrained
#             pass

#     model.to(device)
#     model.eval()

#     return model, tokenizer, device


# def make_local_prediction(model, tokenizer, texts: List[str], device: Optional[str] = None) -> List[Dict[str, Any]]:
#     """
#     Make predictions locally using a model/tokenizer. This is intentionally generic:
#     - If your MultiTaskFinBERT has a forward/predict method, you can adapt this function.
#     """
#     import torch

#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Example: if model is a transformers encoder, just return tokenized inputs as placeholder
#     enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model(**enc)  # may vary by model type

#     # Convert outputs to a serializable placeholder response — user should customize.
#     return [{"raw_output": None, "note": "Customize make_local_prediction to return model outputs"}]

# dashboard/model_utils.py

# Import necessary libraries
import torch

import os
import json
from typing import Any, Dict, List, Tuple, Optional

from huggingface_hub import InferenceApi
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", None)
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "darisdzakwanhoesien/bertesg_tone")


# ======================================================
# Environment Detection (torch optional)
# ======================================================
def torch_available() -> bool:
    """
    Return True if torch is importable in this environment.
    Avoids Streamlit Cloud import errors.
    """
    try:
        import importlib
        spec = importlib.util.find_spec("torch")
        return spec is not None
    except Exception:
        return False


# ======================================================
# HF Inference API (Cloud mode)
# ======================================================
def get_hf_inference_client(repo_id: Optional[str] = None, token: Optional[str] = None) -> InferenceApi:
    repo = repo_id or HF_MODEL_REPO
    tok = token or HF_TOKEN
    return InferenceApi(repo, token=tok)


def make_hf_api_prediction(
    texts: List[str],
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    timeout: int = 120
) -> Any:
    """
    Streamlit Cloud safe fallback
    """
    client = get_hf_inference_client(repo_id, token)
    payload = texts[0] if len(texts) == 1 else texts
    res = client(inputs=payload, params={"use_cache": False}, timeout=timeout)
    return res


# ======================================================
# --- LOCAL MULTITASK MODEL (full torch implementation)
# ======================================================
# Integrated from your provided code
SENTIMENT_LABELS = {0: "positive", 1: "neutral", 2: "negative", 3: "none"}
TONE_LABELS = {0: "Action", 1: "Commitment", 2: "Outcome"}


# ------------------------------------------------------
# MultiTask Model Definition
# ------------------------------------------------------
def get_multitask_model_class():
    """
    Delay the import of torch + define class only in local scenario.
    """
    import torch
    import torch.nn as nn

    class MultiTaskFinBERT(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            hidden = encoder.config.hidden_size
            self.encoder = encoder
            self.dropout = nn.Dropout(0.2)

            self.sentiment_head = nn.Linear(hidden, 4)
            self.tone_head = nn.Linear(hidden, 3)

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids, attention_mask=attention_mask)
            pooled = self.dropout(out.last_hidden_state[:, 0])
            return self.sentiment_head(pooled), self.tone_head(pooled)

    return MultiTaskFinBERT


# ------------------------------------------------------
# Detect correct HF checkpoint file
# ------------------------------------------------------
def find_checkpoint_file(repo_id: str):
    from huggingface_hub import list_repo_files

    files = list_repo_files(repo_id)

    candidates = [
        "pytorch_model.bin",
        "model.safetensors",
        "decoder_model.safetensors",
        "model.ckpt",
        "tf_model.h5",
    ]

    for c in candidates:
        if c in files:
            return c

    for f in files:
        if any(f.endswith(ext) for ext in (".bin", ".safetensors", ".ckpt")):
            return f

    raise FileNotFoundError(f"No weight file found. Repo contains: {files}")


# ------------------------------------------------------
# Load model locally (Scenario A)
# ------------------------------------------------------
def load_local_model_from_hub(
    repo_id: str,
    local_weights_path: Optional[str] = None,
    device: Optional[str] = None
):
    if not torch_available():
        raise RuntimeError("Torch is not available in this environment.")

    import torch
    from transformers import AutoTokenizer, AutoModel
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    encoder = AutoModel.from_pretrained(repo_id)

    MultiTaskFinBERT = get_multitask_model_class()
    model = MultiTaskFinBERT(encoder)

    # choose weight file
    if local_weights_path:
        path = local_weights_path
    else:
        filename = find_checkpoint_file(repo_id)
        path = hf_hub_download(repo_id, filename)
        print(f"[INFO] Using checkpoint file: {filename}")

    # load weights
    if path.endswith(".safetensors"):
        state = load_file(path)
    else:
        state = torch.load(path, map_location=device)

    model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()

    return model, tokenizer, device


# ------------------------------------------------------
# Local Prediction Utility
# ------------------------------------------------------
def make_local_prediction(
    model,
    tokenizer,
    texts: List[str],
    device: Optional[str] = None
) -> List[Dict[str, Any]]:

    import torch

    if isinstance(texts, str):
        texts = [texts]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    input_ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        sent_logits, tone_logits = model(input_ids, mask)
        sent_probs = torch.softmax(sent_logits, dim=1).cpu().numpy()
        tone_probs = torch.softmax(tone_logits, dim=1).cpu().numpy()

    results = []
    for i, txt in enumerate(texts):
        results.append({
            "text": txt,
            "sentiment": SENTIMENT_LABELS[sent_probs[i].argmax()],
            "tone": TONE_LABELS[tone_probs[i].argmax()],
            "sentiment_scores": sent_probs[i].tolist(),
            "tone_scores": tone_probs[i].tolist(),
        })

    return results

def predict_texts(model, tokenizer, texts, device="cpu"):
    if isinstance(texts, str):
        texts = [texts]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        sent_logits, tone_logits = model(input_ids, mask)
        sent_probs = torch.softmax(sent_logits, dim=1).cpu().numpy()
        tone_probs = torch.softmax(tone_logits, dim=1).cpu().numpy()

    results = []
    for i, txt in enumerate(texts):
        results.append({
            "text": txt,
            "sentiment": SENTIMENT_LABELS[sent_probs[i].argmax()],
            "tone": TONE_LABELS[tone_probs[i].argmax()],
            "sentiment_scores": sent_probs[i].tolist(),
            "tone_scores": tone_probs[i].tolist(),
        })

    return results