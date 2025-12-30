import streamlit as st

# -------------------------------------------------------
# SESSION STATE INITIALIZATION (CRITICAL)
# -------------------------------------------------------
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = None

if "filtered_df" not in st.session_state:
    st.session_state["filtered_df"] = None

# -------------------------------------------------------
# Persist filtered dataframe for pages
# -------------------------------------------------------
st.session_state["raw_df"] = raw_df
st.session_state["filtered_df"] = filtered



# # dashboard/app.py
# import os
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import matplotlib.pyplot as plt
# import seaborn as sns
# from dotenv import load_dotenv

# load_dotenv()

# # import light local helpers (model_utils defers heavy imports)
# from model_utils import (
#     torch_available,
#     make_hf_api_prediction,
#     load_local_model_from_hub,
#     make_local_prediction,
# )

# st.set_page_config(page_title="ESG Aspect Dashboard", layout="wide")
# st.title("🔍 ESG Aspect & Ontology Visualization (API-first)")

# # -------------------------
# # Environment detection
# # -------------------------
# IS_LOCAL = torch_available()

# if IS_LOCAL:
#     st.sidebar.success("Environment: Local (torch available)")
# else:
#     st.sidebar.info("Environment: Cloud (torch not available) — using Hugging Face Inference API")

# # -------------------------
# # HF parameters
# # -------------------------
# DEFAULT_HF_REPO = os.getenv("HF_MODEL_REPO", "darisdzakwanhoesien/bertesg_tone")
# hf_repo = st.sidebar.text_input("Hugging Face repo (model)", value=DEFAULT_HF_REPO, help="HF repo id (user/repo). Must be uploaded to Hugging Face Hub.")
# hf_token = st.sidebar.text_input("HF token (optional)", value=os.getenv("HF_TOKEN", ""), type="password")

# # -------------------------
# # Model selection UI
# # -------------------------
# st.sidebar.markdown("### Model source")
# if IS_LOCAL:
#     mode = st.sidebar.radio("Choose model source", ("Local model (dev)", "Hugging Face Inference API"))
# else:
#     mode = "Hugging Face Inference API"
#     st.sidebar.write("Running in cloud, using HF Inference API only.")

# # -------------------------
# # File upload
# # -------------------------
# st.sidebar.markdown("---")
# uploaded_file = st.sidebar.file_uploader("Upload ESG CSV (columns: aspect, aspect_category, ontology_uri, sentiment, tone, sentence)", type=["csv"])
# sample_data_button = st.sidebar.button("Load sample CSV")

# if sample_data_button and uploaded_file is None:
#     # create small sample
#     sample = pd.DataFrame({
#         "aspect": ["pricing", "pricing", "service", "environment"],
#         "aspect_category": ["price", "price", "customer_support", "environmental"],
#         "ontology_uri": ["uri:price", "uri:price", "uri:service", "uri:env"],
#         "sentiment": ["positive", "negative", "neutral", "positive"],
#         "tone": ["formal", "angry", "informative", "optimistic"],
#         "sentence": [
#             "The price is competitive.",
#             "The prices increased unexpectedly.",
#             "Support took long to respond.",
#             "Company reduced emissions."
#         ],
#     })
#     uploaded_file = pd.io.common.BytesIO(sample.to_csv(index=False).encode())

# if uploaded_file is None:
#     st.info("Upload a CSV file to begin or click 'Load sample CSV' in the sidebar.")
#     st.stop()

# # load dataframe
# df = pd.read_csv(uploaded_file)
# df.columns = df.columns.str.strip().str.lower()

# required = {"aspect", "aspect_category", "ontology_uri", "sentiment", "tone", "sentence"}
# missing = required - set(df.columns)
# if missing:
#     st.error(f"Dataset is missing required columns: {missing}")
#     st.stop()

# # basic preview
# st.success("CSV loaded")
# st.dataframe(df.head())

# # -------------------------
# # Model inference area (optional)
# # -------------------------
# st.sidebar.markdown("---")
# st.sidebar.header("Inference (Optional)")

# if mode == "Hugging Face Inference API":
#     st.sidebar.write("Using HF Inference API for predictions.")
#     run_hf = st.sidebar.checkbox("Enable HF inference panel", value=False)
#     if run_hf:
#         text_for_pred = st.text_area("Enter sentence(s) (one per line)", height=150)
#         if st.button("Run HF Inference"):
#             texts = [t.strip() for t in text_for_pred.strip().splitlines() if t.strip()]
#             if len(texts) == 0:
#                 st.warning("Enter at least one sentence.")
#             else:
#                 with st.spinner("Calling HF Inference API..."):
#                     try:
#                         resp = make_hf_api_prediction(texts=texts, repo_id=hf_repo, token=hf_token or None)
#                         st.success("Inference returned")
#                         st.write(resp)
#                     except Exception as e:
#                         st.error(f"HF inference failed: {e}")

# elif mode == "Local model (dev)":
#     st.sidebar.write("Local model mode selected (local PyTorch required).")
#     run_local = st.sidebar.checkbox("Enable local inference", value=False)
#     if run_local:
#         local_weights = st.sidebar.text_input("Local weights path (optional)")
#         if st.sidebar.button("Load local model"):
#             with st.spinner("Loading local model (this may take a while)..."):
#                 try:
#                     model, tokenizer, device = load_local_model_from_hub(hf_repo, local_weights_path=local_weights)
#                     st.sidebar.success("Local model loaded (customize make_local_prediction to use it).")
#                     st.session_state["local_model_loaded"] = True
#                     st.session_state["local_model"] = model
#                     st.session_state["local_tokenizer"] = tokenizer
#                     st.session_state["local_device"] = device
#                 except Exception as e:
#                     st.sidebar.error(f"Loading local model failed: {e}")

#         text_for_pred = st.text_area("Enter sentence(s) (one per line) for local model", height=150)
#         if st.button("Run local model inference"):
#             if not st.session_state.get("local_model_loaded", False):
#                 st.warning("Load the local model first.")
#             else:
#                 texts = [t.strip() for t in text_for_pred.strip().splitlines() if t.strip()]
#                 with st.spinner("Running local prediction..."):
#                     out = make_local_prediction(st.session_state["local_model"], st.session_state["local_tokenizer"], texts, device=st.session_state["local_device"])
#                     st.write(out)

# # -------------------------
# # Dashboard visualizations
# # -------------------------
# st.markdown("## Visualizations")

# # 1 - Aspect category distribution
# st.subheader("1️⃣ Aspect Category Distribution")
# fig1_data = df["aspect_category"].value_counts().reset_index()
# fig1_data.columns = ["aspect_category", "count"]
# fig1 = px.bar(fig1_data, x="aspect_category", y="count", labels={"aspect_category": "Aspect Category", "count": "Count"}, title="Aspect Category Frequency")
# st.plotly_chart(fig1, use_container_width=True)

# # 2 - Ontology URI distribution
# st.subheader("2️⃣ Ontology URI Distribution")
# fig2_data = df["ontology_uri"].value_counts().reset_index()
# fig2_data.columns = ["ontology_uri", "count"]
# fig2 = px.bar(fig2_data, x="ontology_uri", y="count", labels={"ontology_uri": "Ontology URI", "count": "Count"}, title="Ontology URI Frequency")
# st.plotly_chart(fig2, use_container_width=True)

# # 3 - Sentiment by aspect category
# st.subheader("3️⃣ Sentiment by Aspect Category")
# sent_aspect = df.groupby(["aspect_category", "sentiment"]).size().reset_index(name="count")
# fig3 = px.bar(sent_aspect, x="aspect_category", y="count", color="sentiment", barmode="group", title="Sentiment Distribution by Aspect Category")
# st.plotly_chart(fig3, use_container_width=True)

# # 4 - Tone by aspect category
# st.subheader("4️⃣ Tone by Aspect Category")
# tone_aspect = df.groupby(["aspect_category", "tone"]).size().reset_index(name="count")
# fig4 = px.bar(tone_aspect, x="aspect_category", y="count", color="tone", barmode="group", title="Tone Distribution by Aspect Category")
# st.plotly_chart(fig4, use_container_width=True)

# # 5 - Heatmaps
# st.subheader("5️⃣ Aspect Category vs Sentiment / Tone Heatmaps")
# pivot_sent = pd.pivot_table(df, values="sentence", index="aspect_category", columns="sentiment", aggfunc="count", fill_value=0)
# pivot_tone = pd.pivot_table(df, values="sentence", index="aspect_category", columns="tone", aggfunc="count", fill_value=0)

# fig, ax = plt.subplots(1, 2, figsize=(14, 5))
# sns.heatmap(pivot_sent, annot=True, cmap="Blues", ax=ax[0], fmt="d")
# ax[0].set_title("Sentiment Heatmap")
# sns.heatmap(pivot_tone, annot=True, cmap="Greens", ax=ax[1], fmt="d")
# ax[1].set_title("Tone Heatmap")
# st.pyplot(fig)

# # 6 - Wordcloud (optional)
# st.subheader("6️⃣ Wordcloud for sentences")
# from wordcloud import WordCloud
# text = " ".join(df["sentence"].astype(str).tolist())
# if len(text.strip()) > 10:
#     wc = WordCloud(width=800, height=300).generate(text)
#     fig_wc = plt.figure(figsize=(12, 4))
#     plt.imshow(wc, interpolation="bilinear")
#     plt.axis("off")
#     st.pyplot(fig_wc)
# else:
#     st.info("Not enough text to generate wordcloud.")
