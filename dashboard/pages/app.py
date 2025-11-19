# ======================================================
# 🧠 ESG Sentiment & Tone Visualization Dashboard
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt

# Plotly template
pio.templates.default = "plotly_white"

# ----------------------------------------------
# 🎛️ Streamlit Sidebar
# ----------------------------------------------
st.sidebar.title("📊 Dashboard Settings")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

st.title("🧠 ESG Sentiment & Tone Visualization Dashboard")
st.write("Upload your sentiment/tone CSV file to begin.")

# ----------------------------------------------
# 🚨 Exit if no file uploaded
# ----------------------------------------------
if uploaded_file is None:
    st.stop()

# ----------------------------------------------
# 📥 Load Data
# ----------------------------------------------
df = pd.read_csv(uploaded_file)

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

# Required columns
required_cols = {"filename", "sentiment", "tone"}

if not required_cols.issubset(df.columns):
    st.error(f"❌ The dataset must contain these columns: {required_cols}")
    st.stop()

st.success("✅ File loaded successfully!")
st.dataframe(df.head())

# ----------------------------------------------
# 📊 Aggregation
# ----------------------------------------------

# Sentiment summary per document
sentiment_summary = (
    df.groupby("filename")["sentiment"]
    .value_counts()
    .unstack(fill_value=0)
    .reset_index()
)

# Tone summary per document
tone_summary = (
    df.groupby("filename")["tone"]
    .value_counts()
    .unstack(fill_value=0)
    .reset_index()
)

merged = pd.merge(sentiment_summary, tone_summary, on="filename", how="outer").fillna(0)

st.subheader("📄 Aggregated Results per Document")
st.dataframe(merged)

# Detect sentiment & tone columns
sentiment_cols = [c for c in merged.columns if c.lower() in ["positive", "neutral", "negative", "none"]]
tone_cols = [c for c in merged.columns if c.lower() in ["action", "commitment", "outcome"]]

# ----------------------------------------------
# 1️⃣ Sentiment Distribution per Document
# ----------------------------------------------
st.subheader("1️⃣ Sentiment Distribution per Document")

fig1 = px.bar(
    merged,
    x="filename",
    y=sentiment_cols,
    barmode="group",
    title="Sentiment Distribution per Document",
)
st.plotly_chart(fig1, use_container_width=True)

# ----------------------------------------------
# 2️⃣ Tone Distribution per Document
# ----------------------------------------------
st.subheader("2️⃣ Tone Distribution per Document")

fig2 = px.bar(
    merged,
    x="filename",
    y=tone_cols,
    barmode="group",
    title="Tone Distribution per Document",
)
st.plotly_chart(fig2, use_container_width=True)

# ----------------------------------------------
# 3️⃣ Overall Sentiment Pie Chart
# ----------------------------------------------
st.subheader("3️⃣ Overall Sentiment Composition")

sentiment_total = merged[sentiment_cols].sum().reset_index()
sentiment_total.columns = ["sentiment", "count"]

fig3 = px.pie(
    sentiment_total,
    names="sentiment",
    values="count",
    title="Overall Sentiment Composition",
)
st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------------------
# 4️⃣ Overall Tone Pie Chart
# ----------------------------------------------
st.subheader("4️⃣ Overall Tone Composition")

tone_total = merged[tone_cols].sum().reset_index()
tone_total.columns = ["tone", "count"]

fig4 = px.pie(
    tone_total,
    names="tone",
    values="count",
    title="Overall Tone Composition",
)
st.plotly_chart(fig4, use_container_width=True)

# ----------------------------------------------
# 5️⃣ Statistical Summary Table
# ----------------------------------------------
st.subheader("5️⃣ Statistical Summary (Mean ± Std)")

stats_df = merged[sentiment_cols + tone_cols].describe().T[["mean", "std"]]
stats_df["mean"] = stats_df["mean"].round(2)
stats_df["std"] = stats_df["std"].round(2)

st.dataframe(stats_df)

# ----------------------------------------------
# 6️⃣ Correlation Heatmap (Sentiment vs Tone)
# ----------------------------------------------
st.subheader("6️⃣ Sentiment–Tone Correlation Heatmap")

corr = merged[sentiment_cols + tone_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)