import streamlit as st
import pandas as pd
import sys
import os

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.compare_logic import find_missing
from utils.load_hf_file import load_csv_from_hf

# Page config
st.set_page_config(page_title="Dataset Comparison", layout="wide")

st.title("📊 ESG Multi-Dataset Comparison Dashboard")
st.write("Choose data source: Upload, Local, or HuggingFace repository.")

HF_REPO = "darisdzakwanhoesien/esg"

# -------------------------------------------------------
# Helper: Select source & load dataframe
# -------------------------------------------------------
def load_source(selector, uploaded_file, local_path, hf_filename):
    """Load from upload / local / HuggingFace depending on dropdown."""
    
    if selector == "Upload":
        if uploaded_file is None:
            return None, "Upload selected (no file uploaded)"
        df = pd.read_csv(uploaded_file)
        return df, "Uploaded File"

    elif selector == "Local file":
        if not os.path.exists(local_path):
            return None, f"Local file missing: {local_path}"
        df = pd.read_csv(local_path)
        return df, f"Local ({local_path})"

    elif selector == "HuggingFace":
        df, src = load_csv_from_hf(HF_REPO, hf_filename)
        if df is None:
            return None, f"HuggingFace load failed for {hf_filename}"
        return df, src

    return None, "Unknown source"


# -------------------------------------------------------
# Sidebar selectors
# -------------------------------------------------------
st.sidebar.header("📥 Dataset Source Selection")

source_dataset = st.sidebar.selectbox(
    "Dataset.csv source:",
    ["Upload", "Local file", "HuggingFace"],
    index=2  # default HuggingFace
)

source_output = st.sidebar.selectbox(
    "output.csv source:",
    ["Upload", "Local file", "HuggingFace"],
    index=2
)

source_export = st.sidebar.selectbox(
    "output_in_csv.csv source:",
    ["Upload", "Local file", "HuggingFace"],
    index=2
)

uploaded_dataset = st.sidebar.file_uploader("Upload Dataset.csv", type=["csv"])
uploaded_output = st.sidebar.file_uploader("Upload Output.csv", type=["csv"])
uploaded_export = st.sidebar.file_uploader("Upload Output_in_csv.csv", type=["csv"])


# -------------------------------------------------------
# Load each dataset based on dropdown choice
# -------------------------------------------------------
df_dataset, src_dataset = load_source(
    source_dataset,
    uploaded_dataset,
    "data/dataset.csv",
    "Dataset.csv"
)

df_output, src_output = load_source(
    source_output,
    uploaded_output,
    "data/output.csv",
    "output.csv"
)

df_exported, src_exported = load_source(
    source_export,
    uploaded_export,
    "data/output_in_csv.csv",
    "output_in_csv.csv"
)


# -------------------------------------------------------
# Validation
# -------------------------------------------------------
missing = []
if df_dataset is None:
    missing.append("Dataset")
if df_output is None:
    missing.append("Output")
if df_exported is None:
    missing.append("Output in CSV")

if missing:
    st.error(
        "Missing or failed to load: " + ", ".join(missing) +
        "\n\nPlease check your source selection or uploads."
    )
    st.stop()

# Show sources
st.sidebar.success(f"Dataset → **{src_dataset}**")
st.sidebar.success(f"Output → **{src_output}**")
st.sidebar.success(f"Output in CSV → **{src_exported}**")

st.success("✅ All datasets loaded successfully!")


# ==========================================================
# 🔍 Core Logic — Compare Output vs Exported CSV
# ==========================================================
st.header("🔍 Compare Output vs Exported CSV")

missing_rows = find_missing(df_output, df_exported)

st.subheader("📌 Rows in Output but NOT in Output in CSV")
st.write(f"Found **{len(missing_rows)}** missing entries.")

st.dataframe(missing_rows, use_container_width=True)

# Download button
st.download_button(
    "⬇️ Download Missing Rows (CSV)",
    missing_rows.to_csv(index=False),
    file_name="missing_rows.csv",
    mime="text/csv",
)

# Optional previews
with st.expander("📄 Preview: Dataset.csv"):
    st.dataframe(df_dataset.head(), use_container_width=True)

with st.expander("📄 Preview: Output.csv"):
    st.dataframe(df_output.head(), use_container_width=True)

with st.expander("📄 Preview: Output_in_csv.csv"):
    st.dataframe(df_exported.head(), use_container_width=True)
