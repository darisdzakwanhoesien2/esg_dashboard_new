from huggingface_hub import hf_hub_download
import pandas as pd

def load_csv_from_hf(repo_id, filename):
    try:
        local_path = hf_hub_download(repo_id=repo_id, filename=filename)
        df = pd.read_csv(local_path)
        return df, f"HuggingFace ({repo_id}/{filename})"
    except Exception as e:
        print("HF download error:", e)
        return None, None
