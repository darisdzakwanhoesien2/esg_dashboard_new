# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import sys

# ------------------------------------------------------------
# FIX PATHS — Ensure project root is accessible
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

# Import utilities
from dashboard.model_utils import (
    load_local_model_from_hub,
    make_local_prediction,
    make_hf_api_prediction,
    torch_available,
    predict_texts
)

HF_REPO = "darisdzakwanhoesien/bertesg_tone"

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="ESG Model Analytics", page_icon="📈", layout="wide")

st.title("📈 ESG MultiTask FinBERT — Model Analytics Dashboard")
st.write("Explore dataset statistics, sentiment/tone distributions, and model behavior.")

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
@st.cache_data
def load_dataset():
    data_path = os.path.join(PROJECT_ROOT, "data/output_in_csv.csv")
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower().str.strip()
    return df

df = load_dataset()

# ------------------------------------------------------------
# LOAD MODEL (AUTO SWITCH: LOCAL OR CLOUD)
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    ON_STREAMLIT = os.getenv("STREAMLIT_RUNTIME") == "true"

    if ON_STREAMLIT:
        st.warning("🌐 Running on Streamlit Cloud — using HuggingFace Inference API.")
        return ("api", None, None, None)

    if torch_available():
        st.info("💻 Running locally — loading PyTorch model.")
        model, tokenizer, device = load_local_model_from_hub(HF_REPO)
        return ("local", model, tokenizer, device)

    st.warning("⚠️ Torch not available — using HuggingFace Inference API.")
    return ("api", None, None, None)

mode, model, tokenizer, device = load_model()

# ------------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------------
st.sidebar.header("🔍 Base Filters")

aspect_filter = st.sidebar.multiselect(
    "Filter by Aspect Category",
    options=sorted(df["aspect_category"].dropna().unique().tolist()),
)

sentiment_filter = st.sidebar.multiselect(
    "Filter by Sentiment",
    options=sorted(df["sentiment"].dropna().unique().tolist()),
)

tone_filter = st.sidebar.multiselect(
    "Filter by Tone",
    options=sorted(df["tone"].dropna().unique().tolist()),
)

filtered_df = df.copy()
if aspect_filter:
    filtered_df = filtered_df[filtered_df["aspect_category"].isin(aspect_filter)]
if sentiment_filter:
    filtered_df = filtered_df[filtered_df["sentiment"].isin(sentiment_filter)]
if tone_filter:
    filtered_df = filtered_df[filtered_df["tone"].isin(tone_filter)]

# ------------------------------------------------------------
# ADVANCED SUBSETTING (USER-DEFINED PAIRS)
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("🎯 Custom Subset (Single or Multi-Pair)")

subset_mode = st.sidebar.radio(
    "Subset Mode",
    ["Single Pair", "Multi Pair"]
)

if subset_mode == "Single Pair":
    st.sidebar.markdown("Select **one** tone × sentiment pair")
    user_tone = st.sidebar.selectbox("Choose Tone", sorted(df["tone"].dropna().unique()))
    user_sentiment = st.sidebar.selectbox("Choose Sentiment", sorted(df["sentiment"].dropna().unique()))

    subset_df = df[
        (df["tone"] == user_tone) &
        (df["sentiment"] == user_sentiment)
    ].head(50)

else:
    st.sidebar.markdown("Select **multiple** tone × sentiment combinations")
    user_tones = st.sidebar.multiselect("Choose Tone(s)", sorted(df["tone"].dropna().unique()))
    user_sentiments = st.sidebar.multiselect("Choose Sentiment(s)", sorted(df["sentiment"].dropna().unique()))

    if user_tones and user_sentiments:
        subset_df = df[
            (df["tone"].isin(user_tones)) &
            (df["sentiment"].isin(user_sentiments))
        ].head(50)
    else:
        subset_df = pd.DataFrame()

# ------------------------------------------------------------
# DISPLAY SUBSET
# ------------------------------------------------------------
st.markdown("---")
st.subheader("🎯 Custom Subset Results (Max 50 Rows)")

if subset_df.empty:
    st.info("Select tone × sentiment combinations from the sidebar to display subset.")
else:
    st.write(f"Showing **{len(subset_df)} rows**")
    st.dataframe(subset_df, use_container_width=True)

    # ------------------------------------------------------------
    # DOWNLOAD BUTTON
    # ------------------------------------------------------------
    csv_data = subset_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Subset as CSV",
        data=csv_data,
        file_name="subset_selection.csv",
        mime="text/csv"
    )

# st.sidebar.markdown("---")
# st.sidebar.header("🎯 Custom Subset (Single or Multi-Pair)")

# subset_mode = st.sidebar.radio(
#     "Subset Mode",
#     ["Single Pair", "Multi Pair"]
# )

# if subset_mode == "Single Pair":
#     user_tone = st.sidebar.selectbox("Choose Tone", sorted(df["tone"].dropna().unique()))
#     user_sentiment = st.sidebar.selectbox("Choose Sentiment", sorted(df["sentiment"].dropna().unique()))

#     subset_df = df[(df["tone"] == user_tone) & (df["sentiment"] == user_sentiment)].head(50)

# else:
#     user_tones = st.sidebar.multiselect("Choose Tone(s)", sorted(df["tone"].dropna().unique()))
#     user_sentiments = st.sidebar.multiselect("Choose Sentiment(s)", sorted(df["sentiment"].dropna().unique()))

#     if user_tones and user_sentiments:
#         subset_df = df[(df["tone"].isin(user_tones)) & (df["sentiment"].isin(user_sentiments))].head(50)
#     else:
#         subset_df = pd.DataFrame()  # empty

# ------------------------------------------------------------
# DISPLAY SUBSET
# ------------------------------------------------------------
st.markdown("---")
st.subheader("🎯 Custom Subset Results (Max 50 Rows)")

if subset_df.empty:
    st.info("Select one or more tone × sentiment combinations from the sidebar.")
else:
    st.write(f"Showing **{len(subset_df)} rows**")
    st.dataframe(subset_df, use_container_width=True)

# ------------------------------------------------------------
# DATA OVERVIEW
# ------------------------------------------------------------
st.markdown("---")
st.subheader("📄 Dataset Overview (after base filters)")
st.write(f"Showing **{len(filtered_df)} records** after filtering.")
st.dataframe(filtered_df.head(20), use_container_width=True)

# ------------------------------------------------------------
# SENTIMENT DISTRIBUTION
# ------------------------------------------------------------
st.markdown("---")
st.subheader("😊 Sentiment Distribution")

sentiment_counts = filtered_df["sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["sentiment", "count"]

col1, col2 = st.columns(2)

with col1:
    fig1 = px.bar(sentiment_counts, x="sentiment", y="count", color="sentiment", text="count")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.pie(sentiment_counts, names="sentiment", values="count", hole=0.35)
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------
# TONE DISTRIBUTION
# ------------------------------------------------------------
st.markdown("---")
st.subheader("🎭 Tone Distribution")

tone_counts = filtered_df["tone"].value_counts().reset_index()
tone_counts.columns = ["tone", "count"]

col3, col4 = st.columns(2)

with col3:
    fig3 = px.bar(tone_counts, x="tone", y="count", color="tone", text="count")
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    fig4 = px.pie(tone_counts, names="tone", values="count", hole=0.35)
    st.plotly_chart(fig4, use_container_width=True)

# ------------------------------------------------------------
# SENTIMENT × TONE HEATMAP
# ------------------------------------------------------------
st.markdown("---")
st.subheader("🔗 Sentiment × Tone Correlation")

pivot = pd.crosstab(filtered_df["sentiment"], filtered_df["tone"])

fig5, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(pivot, annot=True, cmap="Blues", fmt="d", ax=ax)
st.pyplot(fig5)

# ------------------------------------------------------------
# WORD CLOUD
# ------------------------------------------------------------
st.markdown("---")
st.subheader("☁ Word Cloud")

text_blob = " ".join(filtered_df["sentence"].astype(str).tolist())
wc = WordCloud(width=1200, height=600, background_color="white").generate(text_blob)

fig6, ax2 = plt.subplots(figsize=(12, 6))
ax2.imshow(wc, interpolation="bilinear")
ax2.axis("off")
st.pyplot(fig6)

# ------------------------------------------------------------
# PREDICT SAMPLE
# ------------------------------------------------------------
st.markdown("---")
st.subheader("🧠 Try Model on Random Samples")

if st.button("Generate Predictions for Random 5 Samples"):
    sample_df = filtered_df.sample(5)
    texts = sample_df["sentence"].tolist()

    if mode == "local":
        preds = predict_texts(model, tokenizer, texts, device=device)
    else:
        preds = make_hf_api_prediction(texts)

    result_df = pd.DataFrame([
        {"sentence": p["text"], "sentiment": p["sentiment"], "tone": p["tone"]}
        for p in preds
    ])

    st.write(result_df)

st.success("Analytics loaded successfully!")
