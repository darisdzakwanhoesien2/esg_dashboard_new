import streamlit as st
import pandas as pd

# Utils
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the function
from utils.data_loader import load_csv_uploaded_or_local
from utils.compare_logic import find_missing

# Page config
st.set_page_config(page_title="Dataset Comparison", layout="wide")

st.title("📊 ESG Multi-Dataset Comparison Dashboard")
st.write("Upload or auto-load datasets to analyze differences between `Output` and `Output in CSV`.")

# Sidebar upload section
st.sidebar.header("📥 Upload Datasets (optional)")
uploaded_dataset = st.sidebar.file_uploader("Dataset CSV", type=["csv"])
uploaded_output = st.sidebar.file_uploader("Output CSV", type=["csv"])
uploaded_export = st.sidebar.file_uploader("Output in CSV", type=["csv"])

# Load all datasets using hybrid logic
df_dataset, src_dataset = load_csv_uploaded_or_local(uploaded_dataset, "dataset.csv")
df_output, src_output = load_csv_uploaded_or_local(uploaded_output, "output.csv")
df_exported, src_exported = load_csv_uploaded_or_local(uploaded_export, "output_in_csv.csv")

# Check missing
missing = []
if df_dataset is None:
    missing.append("Dataset")
if df_output is None:
    missing.append("Output")
if df_exported is None:
    missing.append("Output in CSV")

if missing:
    st.warning(f"Missing datasets: {', '.join(missing)}.\n\n"
               "You can upload them in the sidebar or place them inside `dashboard/data/`.")
    st.stop()

# Show load sources
st.sidebar.success(f"Dataset → Loaded from **{src_dataset}**")
st.sidebar.success(f"Output → Loaded from **{src_output}**")
st.sidebar.success(f"Output in CSV → Loaded from **{src_exported}**")

st.success("All datasets loaded successfully!")

# ==========================================================
# 🔍 Core Logic — Find missing Output entries in Exported CSV
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

# Optional: Display previews of loaded datasets
with st.expander("📄 View Loaded Dataset (dataset.csv)"):
    st.dataframe(df_dataset.head(), use_container_width=True)

with st.expander("📄 View Loaded Output (output.csv)"):
    st.dataframe(df_output.head(), use_container_width=True)

with st.expander("📄 View Loaded Output in CSV (output_in_csv.csv)"):
    st.dataframe(df_exported.head(), use_container_width=True)