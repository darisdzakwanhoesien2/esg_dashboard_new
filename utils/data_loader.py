import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def load_csv_uploaded_or_local(upload_file, local_filename):
    """
    Priority:
        1. Uploaded file
        2. Local file inside dashboard/data/
        3. Error
    """
    # Case 1: File uploaded
    if upload_file is not None:
        df = pd.read_csv(upload_file)
        df.columns = df.columns.str.strip().str.lower()
        return df, "uploaded"

    # Case 2: Local file exists
    local_path = os.path.join(DATA_DIR, local_filename)
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        df.columns = df.columns.str.strip().str.lower()
        return df, "local"

    # Case 3: Missing
    return None, "missing"
