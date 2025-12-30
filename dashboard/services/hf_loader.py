import streamlit as st
from transformers import pipeline
from utils.env import get_hf_token

HF_TOKEN = get_hf_token()

@st.cache_resource(show_spinner=False)
def load_pipeline(model_id):
    """
    Safely loads a Hugging Face pipeline.
    Returns (pipeline, None) or (None, error_message)
    """
    try:
        clf = pipeline(
            task="text-classification",
            model=model_id,
            tokenizer=model_id,
            return_all_scores=True,
            token=HF_TOKEN,
            trust_remote_code=True
        )
        return clf, None
    except Exception as e:
        return None, str(e)
