# ======================================================
# 🧠 ESG Sentiment & Tone — Document-Level Dashboard
# ======================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
import json
from pathlib import Path

# ------------------------------------------------
# 🎨 Plotly Theme
# ------------------------------------------------
pio.templates.default = "plotly_white"

# ------------------------------------------------
# 📌 Load Ontologies
# ------------------------------------------------
BASE_DATA_PATH = Path(__file__).resolve().parents[1] / "data"

with open(BASE_DATA_PATH / "sentiment_ontology.json") as f:
    SENTIMENT_ONTOLOGY = json.load(f)

with open(BASE_DATA_PATH / "tone_ontology.json") as f:
    TONE_ONTOLOGY = json.load(f)


def build_alias_map(ontology):
    mapping = {}
    for canonical, meta in ontology.items():
        for alias in meta.get("aliases", []):
            if alias is not None:
                mapping[str(alias).strip().lower()] = canonical
    return mapping


SENTIMENT_MAP = build_alias_map(SENTIMENT_ONTOLOGY)
TONE_MAP = build_alias_map(TONE_ONTOLOGY)


def normalize_sentiment(x):
    if pd.isna(x):
        return "OTHER"
    return SENTIMENT_MAP.get(str(x).strip().lower(), "OTHER")


def normalize_tone(x):
    if pd.isna(x):
        return "OTHER"
    return TONE_MAP.get(str(x).strip().lower(), "OTHER")


# ------------------------------------------------
# 🎛 Sidebar
# ------------------------------------------------
st.sidebar.title("📊 Dashboard Settings")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

st.title("🧠 ESG Sentiment & Tone — Document-Level Analysis")
st.write("Upload a CSV file containing **filename**, **sentiment**, and **tone** columns.")

if uploaded_file is None:
    st.stop()

# ------------------------------------------------
# 📥 Load & Validate Data
# ------------------------------------------------
df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.strip().str.lower()

required_cols = {"filename", "sentiment", "tone"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"❌ Missing required columns: {missing}")
    st.stop()

st.success("✅ File loaded successfully")
st.dataframe(df.head(), use_container_width=True)

# ------------------------------------------------
# 🧹 Normalize FIRST (CRITICAL)
# ------------------------------------------------
df["sentiment_norm"] = df["sentiment"].apply(normalize_sentiment)
df["tone_norm"] = df["tone"].apply(normalize_tone)

# ------------------------------------------------
# 📊 Aggregate per Document (SAFE PREFIXING)
# ------------------------------------------------
sentiment_doc = (
    df.groupby(["filename", "sentiment_norm"])
    .size()
    .unstack(fill_value=0)
    .add_prefix("sent_")
)

tone_doc = (
    df.groupby(["filename", "tone_norm"])
    .size()
    .unstack(fill_value=0)
    .add_prefix("tone_")
)

merged = (
    sentiment_doc
    .join(tone_doc, how="outer")
    .fillna(0)
    .reset_index()
)

# Canonical column orders
SENTIMENT_COLS = [
    "sent_POSITIVE",
    "sent_NEUTRAL",
    "sent_NEGATIVE",
    "sent_OTHER",
]

TONE_COLS = [
    "tone_OUTCOME",
    "tone_ACTION",
    "tone_COMMITMENT",
    "tone_OTHER",
]

# Ensure missing columns exist
for col in SENTIMENT_COLS + TONE_COLS:
    if col not in merged.columns:
        merged[col] = 0

# ------------------------------------------------
# 1️⃣ Sentiment Distribution per Document
# ------------------------------------------------
st.subheader("1️⃣ Sentiment Distribution per Document")

fig1 = px.bar(
    merged,
    x="filename",
    y=SENTIMENT_COLS,
    barmode="group",
    title="Sentiment Distribution per Document",
)
st.plotly_chart(fig1, use_container_width=True)

# ------------------------------------------------
# 2️⃣ Tone Distribution per Document
# ------------------------------------------------
st.subheader("2️⃣ Tone Distribution per Document")

fig2 = px.bar(
    merged,
    x="filename",
    y=TONE_COLS,
    barmode="group",
    title="Tone Distribution per Document",
)
st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------
# 3️⃣ Overall Sentiment Composition
# ------------------------------------------------
st.subheader("3️⃣ Overall Sentiment Composition")

sent_total = merged[SENTIMENT_COLS].sum().reset_index()
sent_total.columns = ["sentiment", "count"]
sent_total["sentiment"] = sent_total["sentiment"].str.replace("sent_", "")

fig3 = px.pie(
    sent_total,
    names="sentiment",
    values="count",
    title="Overall Sentiment Composition",
)
st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------
# 4️⃣ Overall Tone Composition
# ------------------------------------------------
st.subheader("4️⃣ Overall Tone Composition")

tone_total = merged[TONE_COLS].sum().reset_index()
tone_total.columns = ["tone", "count"]
tone_total["tone"] = tone_total["tone"].str.replace("tone_", "")

fig4 = px.pie(
    tone_total,
    names="tone",
    values="count",
    title="Overall Tone Composition",
)
st.plotly_chart(fig4, use_container_width=True)

# ------------------------------------------------
# 5️⃣ Statistical Summary
# ------------------------------------------------
st.subheader("5️⃣ Statistical Summary (Mean ± Std)")

stats_df = merged[SENTIMENT_COLS + TONE_COLS].describe().T[["mean", "std"]]
stats_df["mean"] = stats_df["mean"].round(2)
stats_df["std"] = stats_df["std"].round(2)

st.dataframe(stats_df, use_container_width=True)

# ------------------------------------------------
# 6️⃣ Sentiment–Tone Correlation Heatmap
# ------------------------------------------------
st.subheader("6️⃣ Sentiment–Tone Correlation Heatmap")

corr = merged[SENTIMENT_COLS + TONE_COLS].corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# ------------------------------------------------
# 🧪 Debug Section
# ------------------------------------------------
with st.expander("🧪 Debug: Normalized Columns & Values"):
    st.write("Columns:", merged.columns.tolist())
    st.dataframe(
        df[["filename", "sentiment", "sentiment_norm", "tone", "tone_norm"]].head(30),
        use_container_width=True,
    )
