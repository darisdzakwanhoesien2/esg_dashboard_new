import os

def get_hf_token():
    """
    Returns Hugging Face token if available.
    Priority:
    1. Environment variable HF_TOKEN
    2. None
    """
    return os.getenv("HF_TOKEN")
