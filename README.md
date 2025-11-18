# esg_dashboard_new

# ESG FinBERT Streamlit Dashboard (API-first)

This dashboard is designed to run on Streamlit Cloud (Python 3.13) safely by using Hugging Face Inference API. Local developers can enable local PyTorch/Transformers support by installing `requirements-local.txt` on an appropriate Python environment.

## How it works
- If `torch` is importable, the app enables **Local Model** and **Hugging Face Hub** modes.
- If `torch` is NOT importable (Streamlit Cloud), only **Hugging Face Inference API** mode is available.

## Deploy to Streamlit Cloud
1. Make sure your `requirements.txt` is present (this repo).
2. In Streamlit Cloud, set environment variables (if needed):
   - `HF_TOKEN` (if the HF model is private)
   - `HF_MODEL_REPO` (defaults to `darisdzakwanhoesien/bertesg_tone` in `.env.example`)

## Local development
1. `pip install -r requirements-local.txt` on a machine with a proper Python version and toolchain.
2. Copy `.env.example` to `.env` and fill `HF_TOKEN` if needed.

## Notes
- The app uses `huggingface_hub.InferenceApi` in cloud mode to avoid heavy local binary installs.
- For local inference, `model_utils.py` contains functions to load local models and run predictions.



esg-dashboard/
├─ dashboard/
│  ├─ app.py                 # main Streamlit app (use this)
│  ├─ model_utils.py         # model load / inference helpers (hybrid)
│  ├─ finbert_model.py       # MultiTaskFinBERT model class skeleton
│  ├─ assets/
│  │   └─ ...                # images/static assets if needed
├─ requirements_streamlit.txt   # for Streamlit Cloud (NO torch)
├─ requirements_local.txt       # for local dev (with torch, transformers)
└─ README.md
