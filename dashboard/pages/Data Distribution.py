import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio
import json
from pathlib import Path

# ------------------------------------------------
# 🎨 Plotly Theme
# ------------------------------------------------
pio.templates.default = "plotly_white"

# ------------------------------------------------
# 📌 Page Title
# ------------------------------------------------
st.title("🔍 Aspect & Ontology Visualization Dashboard")
st.write(
    "Ontology-driven analysis of aspect categories, ontology URIs, "
    "sentiment, and tone at sentence level."
)

# ------------------------------------------------
# 📥 Sidebar — File Upload
# ------------------------------------------------
uploaded_file = st.sidebar.file_uploader(
    "Upload ESG CSV", type=["csv"], key="aspect_file"
)

if uploaded_file is None:
    st.info("⬅️ Upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

# ------------------------------------------------
# 🔐 Normalize Column Names
# ------------------------------------------------
df.columns = df.columns.str.strip().str.lower()

REQUIRED_COLS = {
    "aspect",
    "aspect_category",
    "ontology_uri",
    "sentiment",
    "tone",
    "sentence",
}

missing = REQUIRED_COLS - set(df.columns)
if missing:
    st.error(f"❌ Missing required columns: {missing}")
    st.stop()

st.success("✅ File loaded successfully")
st.dataframe(df.head(), use_container_width=True)

# =================================================
# 📚 LOAD ONTOLOGIES
# =================================================
BASE_DATA_PATH = Path(__file__).resolve().parents[1] / "data"

# -----------------------------
# Aspect Ontology
# -----------------------------
with open(BASE_DATA_PATH / "aspect_category_ontology.json") as f:
    ASPECT_ONTOLOGY = json.load(f)

ASPECT_ALIAS_MAP = {
    str(alias).strip().upper(): canonical
    for canonical, meta in ASPECT_ONTOLOGY.items()
    for alias in meta.get("aliases", [])
    if alias is not None
}

def normalize_aspect_category(value):
    if pd.isna(value):
        return "OTHER"
    return ASPECT_ALIAS_MAP.get(str(value).strip().upper(), "OTHER")

def aspect_label(canonical):
    return ASPECT_ONTOLOGY.get(canonical, {}).get("label", canonical)

# -----------------------------
# Sentiment Ontology
# -----------------------------
with open(BASE_DATA_PATH / "sentiment_ontology.json") as f:
    SENTIMENT_ONTOLOGY = json.load(f)

SENTIMENT_ALIAS_MAP = {
    str(alias).strip().lower(): canonical
    for canonical, meta in SENTIMENT_ONTOLOGY.items()
    for alias in meta.get("aliases", [])
    if alias is not None
}

def normalize_sentiment(value):
    if pd.isna(value):
        return "OTHER"
    return SENTIMENT_ALIAS_MAP.get(str(value).strip().lower(), "OTHER")

def sentiment_label(canonical):
    return SENTIMENT_ONTOLOGY.get(canonical, {}).get("label", canonical)

# -----------------------------
# Tone Ontology  ✅ NEW
# -----------------------------
with open(BASE_DATA_PATH / "tone_ontology.json") as f:
    TONE_ONTOLOGY = json.load(f)

TONE_ALIAS_MAP = {
    str(alias).strip().lower(): canonical
    for canonical, meta in TONE_ONTOLOGY.items()
    for alias in meta.get("aliases", [])
    if alias is not None
}

def normalize_tone(value):
    if pd.isna(value):
        return "OTHER"
    return TONE_ALIAS_MAP.get(str(value).strip().lower(), "OTHER")

def tone_label(canonical):
    return TONE_ONTOLOGY.get(canonical, {}).get("label", canonical)

# =================================================
# 🧹 APPLY NORMALIZATION
# =================================================
df["aspect_category_raw"] = df["aspect_category"]
df["aspect_category_norm"] = df["aspect_category"].apply(normalize_aspect_category)
df["aspect_category_label"] = df["aspect_category_norm"].apply(aspect_label)

df["sentiment_raw"] = df["sentiment"]
df["sentiment_norm"] = df["sentiment"].apply(normalize_sentiment)
df["sentiment_label"] = df["sentiment_norm"].apply(sentiment_label)

df["tone_raw"] = df["tone"]
df["tone_norm"] = df["tone"].apply(normalize_tone)
df["tone_label"] = df["tone_norm"].apply(tone_label)

ESG_ORDER = ["E", "S", "G", "E-S", "E-G", "S-G", "E-S-G", "OTHER"]
SENTIMENT_ORDER = ["POSITIVE", "NEUTRAL", "NEGATIVE", "OTHER"]
TONE_ORDER = ["OUTCOME", "ACTION", "COMMITMENT", "OTHER"]

# =================================================
# 1️⃣ Aspect Category Distribution
# =================================================
st.subheader("1️⃣ Aspect Category Distribution")

fig1_data = (
    df["aspect_category_norm"]
    .value_counts()
    .reindex(ESG_ORDER, fill_value=0)
    .reset_index()
)

fig1_data.columns = ["aspect_category_norm", "count"]
fig1_data["label"] = fig1_data["aspect_category_norm"].apply(aspect_label)

fig1 = px.bar(
    fig1_data,
    x="label",
    y="count",
    text="count",
    title="Aspect Category Frequency",
)

fig1.update_layout(xaxis_tickangle=-30)
st.plotly_chart(fig1, use_container_width=True)

# =================================================
# 2️⃣ Ontology URI Distribution
# =================================================
st.subheader("2️⃣ Ontology URI Distribution")

fig2_data = (
    df["ontology_uri"]
    .fillna("UNKNOWN")
    .value_counts()
    .head(30)
    .reset_index()
)

fig2_data.columns = ["ontology_uri", "count"]

fig2 = px.bar(
    fig2_data,
    x="ontology_uri",
    y="count",
    title="Top Ontology URI Frequency",
)

fig2.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig2, use_container_width=True)

# =================================================
# 3️⃣ Sentiment Distribution by Aspect Category
# =================================================
st.subheader("3️⃣ Sentiment Distribution by Aspect Category")

sent_aspect = (
    df.groupby(["aspect_category_norm", "sentiment_norm"])
    .size()
    .reset_index(name="count")
)

sent_aspect["aspect_label"] = sent_aspect["aspect_category_norm"].apply(aspect_label)
sent_aspect["sentiment_label"] = sent_aspect["sentiment_norm"].apply(sentiment_label)

fig3 = px.bar(
    sent_aspect,
    x="aspect_label",
    y="count",
    color="sentiment_label",
    barmode="group",
    title="Sentiment Distribution by Aspect Category",
)

fig3.update_layout(xaxis_tickangle=-30)
st.plotly_chart(fig3, use_container_width=True)

# =================================================
# 4️⃣ Tone Distribution by Aspect Category ✅ FIXED
# =================================================
st.subheader("4️⃣ Tone Distribution by Aspect Category")

tone_aspect = (
    df.groupby(["aspect_category_norm", "tone_norm"])
    .size()
    .reset_index(name="count")
)

tone_aspect["aspect_label"] = tone_aspect["aspect_category_norm"].apply(aspect_label)
tone_aspect["tone_label"] = tone_aspect["tone_norm"].apply(tone_label)

fig4 = px.bar(
    tone_aspect,
    x="aspect_label",
    y="count",
    color="tone_label",
    barmode="group",
    title="Tone Distribution by Aspect Category",
)

fig4.update_layout(xaxis_tickangle=-30)
st.plotly_chart(fig4, use_container_width=True)

# =================================================
# 5️⃣ Heatmaps
# =================================================
st.subheader("5️⃣ Aspect Category × Sentiment / Tone Heatmaps")

pivot_sent = pd.pivot_table(
    df,
    values="sentence",
    index="aspect_category_norm",
    columns="sentiment_label",
    aggfunc="count",
    fill_value=0,
).reindex(ESG_ORDER)

pivot_tone = pd.pivot_table(
    df,
    values="sentence",
    index="aspect_category_norm",
    columns="tone_label",
    aggfunc="count",
    fill_value=0,
).reindex(ESG_ORDER)

fig, ax = plt.subplots(1, 2, figsize=(16, 5))

sns.heatmap(pivot_sent, annot=True, cmap="Blues", ax=ax[0])
ax[0].set_title("Sentiment Heatmap")

sns.heatmap(pivot_tone, annot=True, cmap="Greens", ax=ax[1])
ax[1].set_title("Tone Heatmap")

st.pyplot(fig)

# =================================================
# 📤 JSON EXPORTS
# =================================================
st.subheader("📤 Export Normalized JSON Annotations")

# Aspect JSON
aspect_json = (
    df["aspect_category_norm"]
    .value_counts()
    .reindex(ESG_ORDER, fill_value=0)
    .reset_index()
)
aspect_json.columns = ["aspect", "count"]
aspect_json["label"] = aspect_json["aspect"].apply(aspect_label)

st.download_button(
    "Download Aspect Summary (JSON)",
    json.dumps(aspect_json.to_dict(orient="records"), indent=2),
    "aspect_category_summary.json",
    "application/json",
)

# Sentiment JSON
sentiment_json = (
    df["sentiment_norm"]
    .value_counts()
    .reindex(SENTIMENT_ORDER, fill_value=0)
    .reset_index()
)
sentiment_json.columns = ["sentiment", "count"]
sentiment_json["label"] = sentiment_json["sentiment"].apply(sentiment_label)

st.download_button(
    "Download Sentiment Summary (JSON)",
    json.dumps(sentiment_json.to_dict(orient="records"), indent=2),
    "sentiment_summary.json",
    "application/json",
)

# Tone JSON ✅ NEW
tone_json = (
    df["tone_norm"]
    .value_counts()
    .reindex(TONE_ORDER, fill_value=0)
    .reset_index()
)
tone_json.columns = ["tone", "count"]
tone_json["label"] = tone_json["tone"].apply(tone_label)

st.download_button(
    "Download Tone Summary (JSON)",
    json.dumps(tone_json.to_dict(orient="records"), indent=2),
    "tone_summary.json",
    "application/json",
)

# =================================================
# 🧪 DEBUG VIEW
# =================================================
with st.expander("🧪 Debug: Raw vs Normalized Labels"):
    st.dataframe(
        df[
            [
                "aspect_category_raw",
                "aspect_category_norm",
                "sentiment_raw",
                "sentiment_norm",
                "tone_raw",
                "tone_norm",
                "sentence",
            ]
        ].head(30),
        use_container_width=True,
    )
