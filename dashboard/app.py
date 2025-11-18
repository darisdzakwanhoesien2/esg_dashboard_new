# dashboard/app.py
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from model_utils import (
    load_local_model,
    make_local_prediction,
    make_hf_api_prediction,
    hf_repo_list,  # helper to list repo (optional)
)
st.set_page_config(layout="wide", page_title="ESG FinBERT Dashboard")

# ------------------------------------------------
# Sidebar - Model Source & Input
# ------------------------------------------------
st.sidebar.title("🔧 Model Source")
model_source = st.sidebar.radio("Load model from:", ("HuggingFace Hub", "Local Directory"))

# HuggingFace Hub selection
hf_repo = None
if model_source == "HuggingFace Hub":
    st.sidebar.write("Select HuggingFace Repo (public/private).")
    # Provide a text input or a dropdown if you have known repos
    hf_repo = st.sidebar.text_input("HuggingFace repo id (user/repo)", value="your-username/your-model")
    hf_token = st.sidebar.text_input("HF token (optional, for private models)", value="", type="password")

# Local directory
local_dir = None
if model_source == "Local Directory":
    st.sidebar.write("Point to a local directory containing a model (config + pytorch_model.bin or safetensors).")
    local_dir = st.sidebar.text_input("Local model folder path", value="")

# ------------------------------------------------
# File upload (shared across pages)
# ------------------------------------------------
st.sidebar.title("📥 Data")
uploaded_file = st.sidebar.file_uploader("Upload ESG CSV (sentence,aspect,aspect_category,ontology_uri,sentiment,tone)", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV file to proceed.")
    st.stop()

df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.strip().str.lower()
required = {"sentence", "aspect", "aspect_category", "ontology_uri", "sentiment", "tone"}
if not required.issubset(df.columns):
    st.error(f"Dataset must contain columns: {required}")
    st.stop()

# ------------------------------------------------
# Model loading area
# ------------------------------------------------
st.sidebar.title("🔁 Model Loader")
loaded_model = None
tokenizer = None
device = None
hf_api_client = None
load_msg = st.sidebar.empty()

if model_source == "HuggingFace Hub":
    if not hf_repo:
        load_msg.info("Enter a HuggingFace repo id to load the model via HF Inference API.")
    else:
        try:
            load_msg.info(f"Loading HF model: {hf_repo} …")
            # For HF hub we will not load heavy tokenizers or torch.
            # Instead we will use the inference API wrapper when predicting.
            # To check repo exists you may optionally call hf_repo_list()
            # (not strictly necessary here)
            hf_api_client = {
                "repo_id": hf_repo,
                "token": hf_token or None
            }
            load_msg.success(f"Configured HF repo: {hf_repo}")
        except Exception as e:
            load_msg.error(f"Failed to configure HF repo: {e}")

elif model_source == "Local Directory":
    if not local_dir:
        load_msg.info("Enter path to a local model directory.")
    else:
        try:
            load_msg.info(f"Loading local model from {local_dir} …")
            loaded_model, tokenizer, device = load_local_model(local_dir)
            load_msg.success("Local model loaded successfully.")
        except Exception as e:
            load_msg.error(f"Failed to load local model: {e}")
            st.sidebar.exception(e)

# ------------------------------------------------
# Main UI
# ------------------------------------------------
st.title("🔍 Aspect & Ontology Visualization Dashboard")
st.write("Analyze aspect categories, ontology URIs, sentiment and tone at sentence level.")

st.success("File loaded successfully!")
st.dataframe(df.head(), use_container_width=True)

# Wordcloud (right column)
col1, col2 = st.columns([1, 2])
with col2:
    st.subheader("Wordcloud")
    text = " ".join(df["sentence"].astype(str).values)
    if text.strip():
        wc = WordCloud(width=800, height=250, background_color="white").generate(text)
        fig_wc, ax = plt.subplots(figsize=(12, 3))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wc)
    else:
        st.info("No text available for wordcloud.")

# Try model on random samples
st.markdown("### 🧠 Try Model on Random Samples")
if st.button("Generate Predictions for Random 5 Samples"):
    sample = df["sentence"].dropna().sample(min(5, len(df))).tolist()
    results = []
    for s in sample:
        if model_source == "HuggingFace Hub" and hf_api_client:
            pred = make_hf_api_prediction(s, repo_id=hf_api_client["repo_id"], token=hf_api_client["token"])
        elif model_source == "Local Directory" and loaded_model is not None:
            pred = make_local_prediction(s, loaded_model, tokenizer, device)
        else:
            pred = {"error": "No model configured."}
        results.append({"sentence": s, **pred})
    res_df = pd.DataFrame(results)
    st.table(res_df)

# Left column analytics (charts)
with col1:
    st.subheader("Filters")
    aspect_options = ["All"] + sorted(df["aspect_category"].dropna().unique().tolist())
    sel_aspect = st.selectbox("Filter by Aspect Category", aspect_options)
    sel_sentiment = st.selectbox("Filter by Sentiment", ["All"] + sorted(df["sentiment"].dropna().unique().tolist()))
    sel_tone = st.selectbox("Filter by Tone", ["All"] + sorted(df["tone"].dropna().unique().tolist()))

filtered = df.copy()
if sel_aspect != "All":
    filtered = filtered[filtered["aspect_category"] == sel_aspect]
if sel_sentiment != "All":
    filtered = filtered[filtered["sentiment"] == sel_sentiment]
if sel_tone != "All":
    filtered = filtered[filtered["tone"] == sel_tone]

st.subheader("1️⃣ Aspect Category Distribution")
fig1 = px.bar(filtered["aspect_category"].value_counts().reset_index().rename(columns={"index":"aspect_category","aspect_category":"count"}),
              x="aspect_category", y="count", title="Aspect Category Frequency")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("2️⃣ Ontology URI Distribution")
fig2 = px.bar(filtered["ontology_uri"].value_counts().reset_index().rename(columns={"index":"ontology_uri","ontology_uri":"count"}),
              x="ontology_uri", y="count", title="Ontology URI Frequency")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("3️⃣ Sentiment by Aspect Category")
sent_aspect = filtered.groupby(["aspect_category","sentiment"]).size().reset_index(name="count")
fig3 = px.bar(sent_aspect, x="aspect_category", y="count", color="sentiment", barmode="group", title="Sentiment Distribution by Aspect Category")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("4️⃣ Tone by Aspect Category")
tone_aspect = filtered.groupby(["aspect_category","tone"]).size().reset_index(name="count")
fig4 = px.bar(tone_aspect, x="aspect_category", y="count", color="tone", barmode="group", title="Tone Distribution by Aspect Category")
st.plotly_chart(fig4, use_container_width=True)

st.subheader("5️⃣ Heatmap: Aspect Category vs Sentiment/Tone")
pivot_sent = pd.pivot_table(filtered, values="sentence", index="aspect_category", columns="sentiment", aggfunc="count", fill_value=0)
pivot_tone = pd.pivot_table(filtered, values="sentence", index="aspect_category", columns="tone", aggfunc="count", fill_value=0)
fig, ax = plt.subplots(1,2, figsize=(14,5))
sns.heatmap(pivot_sent, annot=True, cmap="Blues", ax=ax[0])
ax[0].set_title("Sentiment Heatmap")
sns.heatmap(pivot_tone, annot=True, cmap="Greens", ax=ax[1])
ax[1].set_title("Tone Heatmap")
st.pyplot(fig)

st.write("Analytics loaded successfully!")
