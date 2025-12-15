import streamlit as st
import pandas as pd
import json
import os
import re

# -------------------------------------------------------
# Page Config
# -------------------------------------------------------
st.set_page_config(page_title="Parsed ESG JSON Dashboard", layout="wide")
st.title("📊 ESG Parsed Sentence-Level Dashboard")

# -------------------------------------------------------
# Load CSV
# -------------------------------------------------------
DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "data_output.csv")
)

st.caption(f"Using data: `{DATA_PATH}`")

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

try:
    raw_df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"❌ Failed to load CSV at: {DATA_PATH}\n\n{e}")
    st.stop()

# =======================================================
# ROBUST JSON PARSING (NEW)
# =======================================================

def extract_json_block(text):
    """
    Extract the first JSON array or object from messy LLM output.
    """
    if not isinstance(text, str):
        return None

    match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(1))
    except Exception:
        return None


def normalize_json(obj):
    """
    Normalize JSON into flat list of dicts.
    Handles dicts, lists, nested lists.
    """
    if obj is None:
        return []

    if isinstance(obj, dict):
        return [obj]

    if isinstance(obj, list):
        flat = []
        for item in obj:
            flat.extend(normalize_json(item))
        return flat

    return []


def is_valid_esg_object(d):
    """
    Minimal ESG validation.
    """
    return (
        isinstance(d, dict)
        and "sentence" in d
        and "aspect" in d
    )


def parse_esg_json(text):
    """
    End-to-end robust ESG JSON parser.
    """
    raw = extract_json_block(text)
    normalized = normalize_json(raw)
    validated = [x for x in normalized if is_valid_esg_object(x)]
    return validated


# -------------------------------------------------------
# Parse JSON annotations (UPDATED)
# -------------------------------------------------------
@st.cache_data
def parse_annotations(df):
    df = df.copy()
    df["parsed"] = df["text"].apply(parse_esg_json)

    exploded = df.explode("parsed", ignore_index=True)
    parsed_df = pd.json_normalize(exploded["parsed"])

    meta_cols = [c for c in df.columns if c != "parsed"]
    meta = exploded[meta_cols].reset_index(drop=True)

    full = pd.concat([meta, parsed_df], axis=1)
    return full


df = parse_annotations(raw_df)

st.success(f"Parsed **{len(df)}** ESG sentence records")

# -------------------------------------------------------
# Helper: Parse provider
# -------------------------------------------------------
def parse_provider(m):
    if isinstance(m, str) and "/" in m:
        return m.split("/")[0]
    return "unknown"

df["provider"] = df["model"].apply(parse_provider)

# -------------------------------------------------------
# Helper: Ensure pivot contains ALL models
# -------------------------------------------------------
def ensure_all_models(reference_df, pivot):
    all_models = sorted(reference_df["model"].dropna().unique())
    for m in all_models:
        if m not in pivot.columns:
            pivot[m] = None
    return pivot[all_models]

# -------------------------------------------------------
# Helper: completeness scoring
# -------------------------------------------------------
def model_completeness(df_pdf, df_page):
    expected = sorted(df_pdf["model"].dropna().unique())
    present = sorted(df_page["model"].dropna().unique())
    missing = set(expected) - set(present)

    score = len(present) / len(expected) if expected else 1.0

    return {
        "expected": expected,
        "present": present,
        "missing": sorted(missing),
        "missing_count": len(missing),
        "present_count": len(present),
        "total": len(expected),
        "score": score
    }

# -------------------------------------------------------
# Sidebar Filters
# -------------------------------------------------------
st.sidebar.header("🔍 Filters")

def make_multiselect(label, col):
    if col not in df.columns:
        return None
    vals = sorted(df[col].dropna().unique())
    return st.sidebar.multiselect(label, vals, default=vals)

aspect_cats = make_multiselect("Aspect Category", "aspect_category")
sentiments = make_multiselect("Sentiment", "sentiment")
tones = make_multiselect("Tone", "tone")
materialities = make_multiselect("Materiality", "materiality")
stakeholders = make_multiselect("Stakeholder", "stakeholder")
value_chain_stage = make_multiselect("Value Chain Stage", "value_chain_stage")
time_horizon = make_multiselect("Time Horizon", "time_horizon")

if "confidence" in df.columns:
    conf_range = st.sidebar.slider(
        "Confidence Range",
        0.0, 1.0,
        (float(df["confidence"].min()), float(df["confidence"].max())),
        0.01,
    )
else:
    conf_range = None

filtered = df.copy()

def apply_filter(col, values):
    global filtered
    if values and col in filtered.columns:
        filtered = filtered[filtered[col].isin(values)]

apply_filter("aspect_category", aspect_cats)
apply_filter("sentiment", sentiments)
apply_filter("tone", tones)
apply_filter("materiality", materialities)
apply_filter("stakeholder", stakeholders)
apply_filter("value_chain_stage", value_chain_stage)
apply_filter("time_horizon", time_horizon)

if conf_range:
    lo, hi = conf_range
    filtered = filtered[(filtered["confidence"] >= lo) & (filtered["confidence"] <= hi)]

st.caption(f"Showing **{len(filtered)}** sentences after filtering.")

# -------------------------------------------------------
# Tabs
# -------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Distributions",
    "📌 Aspects",
    "📄 Sentence Table",
    "🤖 LLM Model Comparison",
    "LLM Breakdown",
    "🧮 Model Coverage",
    "📦 Raw JSON View"
])

# -------------------------------------------------------
# TAB 1 — Distributions
# -------------------------------------------------------
with tab1:
    st.subheader("Sentiment Distribution")
    st.bar_chart(filtered["sentiment"].value_counts())

    st.subheader("Aspect Category Distribution")
    st.bar_chart(filtered["aspect_category"].value_counts())

# -------------------------------------------------------
# TAB 2 — Aspects
# -------------------------------------------------------
with tab2:
    st.subheader("Top Aspects")
    if "aspect" in filtered:
        n = st.slider("Show Top N", 3, 30, 10)
        topA = filtered["aspect"].value_counts().head(n)
        st.bar_chart(topA)
        st.dataframe(topA.rename("count"))

# -------------------------------------------------------
# TAB 3 — Sentence Table
# -------------------------------------------------------
with tab3:
    st.subheader("Full Sentence Table")
    wanted = [
        "sentence","aspect","aspect_category","sentiment","sentiment_score",
        "tone","materiality","stakeholder","impact_level","time_horizon",
        "filename","page_number","model"
    ]
    show_cols = [c for c in wanted if c in filtered.columns]
    st.dataframe(filtered[show_cols], use_container_width=True)

# -------------------------------------------------------
# TAB 4 — Model Comparison (WITH MARKDOWN VISUALIZATION)
# -------------------------------------------------------
with tab4:
    st.subheader("🤖 LLM Model Comparison (Page-Level)")

    # ---------------------------------------------------
    # File & Page Selection
    # ---------------------------------------------------
    filenames = sorted(filtered["filename"].unique())
    selected_file = st.selectbox("Filename", filenames, key="mc_file")

    pages = sorted(
        filtered[filtered["filename"] == selected_file]["page_number"].unique()
    )
    selected_page = st.selectbox("Page Number", pages, key="mc_page")

    subset = filtered[
        (filtered["filename"] == selected_file) &
        (filtered["page_number"] == selected_page)
    ]

    if subset.empty:
        st.warning("No data for this file & page.")
        st.stop()

    # ---------------------------------------------------
    # Model Completeness
    # ---------------------------------------------------
    df_pdf = filtered[filtered["filename"] == selected_file]
    comp = model_completeness(df_pdf, subset)

    st.metric(
        "Model Completeness",
        f"{comp['score']*100:.1f}%",
        help=f"Present: {comp['present_count']} / {comp['total']}"
    )

    if comp["missing_count"] > 0:
        st.warning(
            f"Missing models: {', '.join(comp['missing'])}"
        )

    # ---------------------------------------------------
    # MARKDOWN CONTEXT (SOURCE OF TRUTH)
    # ---------------------------------------------------
    st.markdown("## 📄 Source Text (Same for All Models)")

    # Take FIRST row — guaranteed same per file+page
    row0 = subset.iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧾 markdown_full")
        if "markdown_full" in row0 and pd.notna(row0["markdown_full"]):
            st.markdown(row0["markdown_full"])
        else:
            st.info("No markdown_full available.")

    with col2:
        st.markdown("### ✂️ cleaned_markdown")
        if "cleaned_markdown" in row0 and pd.notna(row0["cleaned_markdown"]):
            st.markdown(row0["cleaned_markdown"])
        else:
            st.info("No cleaned_markdown available.")

    # ---------------------------------------------------
    # Sentence-Level Model Comparison
    # ---------------------------------------------------
    st.markdown("## 🔍 Sentence-Level Comparison Across Models")

    pivot = subset.pivot_table(
        index="sentence",
        columns="model",
        values="sentiment",
        aggfunc="first"
    )

    pivot = ensure_all_models(df_pdf, pivot)
    st.dataframe(pivot, use_container_width=True)

    # ---------------------------------------------------
    # Sentence Existence Check (PER SENTENCE)
    # ---------------------------------------------------
    st.markdown("## ✅ Sentence Presence Check (Against Source Text)")

    markdown_full = str(row0.get("markdown_full", "") or "")
    cleaned_markdown = str(row0.get("cleaned_markdown", "") or "")
    raw_text = str(row0.get("text", "") or "")

    def sentence_presence(sentence):
        return {
            "in_text": sentence in raw_text,
            "in_markdown_full": sentence in markdown_full,
            "in_cleaned_markdown": sentence in cleaned_markdown
        }

    presence_rows = []
    for sent in pivot.index:
        presence = sentence_presence(sent)
        presence_rows.append({
            "sentence": sent,
            "in_text": presence["in_text"],
            "in_markdown_full": presence["in_markdown_full"],
            "in_cleaned_markdown": presence["in_cleaned_markdown"],
            "found_anywhere": any(presence.values())
        })

    presence_df = pd.DataFrame(presence_rows)

    st.dataframe(
        presence_df,
        use_container_width=True
    )

    # ---------------------------------------------------
    # Highlight Missing Sentences
    # ---------------------------------------------------
    missing = presence_df[~presence_df["found_anywhere"]]

    if not missing.empty:
        st.warning(
            f"⚠️ {len(missing)} sentences are NOT found in markdown_full / cleaned_markdown / raw text."
        )
        st.dataframe(
            missing[["sentence"]],
            use_container_width=True
        )
    else:
        st.success("✅ All extracted sentences exist in the source text.")


# # -------------------------------------------------------
# # TAB 4 — Model Comparison (Improved)
# # -------------------------------------------------------
# with tab4:
#     st.subheader("LLM Model Comparison")

#     filenames = sorted(filtered["filename"].unique())
#     selected_file = st.selectbox("Filename", filenames)

#     pages = sorted(filtered[filtered["filename"] == selected_file]["page_number"].unique())
#     selected_page = st.selectbox("Page", pages)

#     subset = filtered[
#         (filtered["filename"] == selected_file) &
#         (filtered["page_number"] == selected_page)
#     ]

#     # Model completeness
#     comp = model_completeness(filtered[filtered["filename"] == selected_file], subset)
#     st.metric("Model Completeness", f"{comp['score']*100:.1f}%")

#     # Create pivot table for sentence comparison
#     pivot = subset.pivot_table(
#         index="sentence",
#         columns="model",
#         values="sentiment",
#         aggfunc="first"
#     )

#     # Ensure all models are present
#     pivot = ensure_all_models(filtered[filtered["filename"] == selected_file], pivot)

#     # Display sentence-level comparison
#     st.subheader("🔍 Sentence-Level Comparison")
#     st.dataframe(pivot, use_container_width=True)

#     # Combine sentences into a single string to check for existence in text/markdown
#     combined_sentences = pivot.index.tolist()

#     # Check if combined sentences exist in 'text', 'markdown_full', or 'cleaned_markdown'
#     def check_sentence_existence(sentence, row):
#         """
#         Check if the combined sentence exists in the provided columns.
#         """
#         text = row['text']
#         markdown_full = row.get('markdown_full', '')
#         cleaned_markdown = row.get('cleaned_markdown', '')

#         # Combine the sentences to check if they exist in any of the relevant columns
#         combined_text = ' '.join(combined_sentences)

#         if any([
#             combined_text in text,
#             combined_text in markdown_full,
#             combined_text in cleaned_markdown
#         ]):
#             return True
#         return False

#     # Check for each sentence's existence in the columns
#     sentence_existence_results = []
#     for sentence in combined_sentences:
#         sentence_found = False
#         for idx, row in subset.iterrows():
#             if check_sentence_existence(sentence, row):
#                 sentence_found = True
#                 break
#         sentence_existence_results.append((sentence, sentence_found))

#     # Display results of sentence comparison with existence check
#     st.subheader("📄 Sentence Existence Check")
#     existence_df = pd.DataFrame(sentence_existence_results, columns=["Sentence", "Exists in Data (text/markdown)"])
#     st.dataframe(existence_df)

#     # Show warnings for missing sentences
#     missing_sentences = existence_df[existence_df["Exists in Data (text/markdown)"] == False]
#     if not missing_sentences.empty:
#         st.warning(f"⚠️ {len(missing_sentences)} sentences are **missing** in the data (text or markdown columns).")
#         st.dataframe(missing_sentences)


# # -------------------------------------------------------
# # TAB 4 — Model Comparison
# # -------------------------------------------------------
# with tab4:
#     st.subheader("LLM Model Comparison")

#     filenames = sorted(filtered["filename"].unique())
#     selected_file = st.selectbox("Filename", filenames)

#     pages = sorted(filtered[filtered["filename"] == selected_file]["page_number"].unique())
#     selected_page = st.selectbox("Page", pages)

#     subset = filtered[
#         (filtered["filename"] == selected_file) &
#         (filtered["page_number"] == selected_page)
#     ]

#     comp = model_completeness(filtered[filtered["filename"] == selected_file], subset)
#     st.metric("Model Completeness", f"{comp['score']*100:.1f}%")

#     pivot = subset.pivot_table(
#         index="sentence",
#         columns="model",
#         values="sentiment",
#         aggfunc="first"
#     )

#     pivot = ensure_all_models(filtered[filtered["filename"] == selected_file], pivot)
#     st.dataframe(pivot, use_container_width=True)

# -------------------------------------------------------
# TAB 5 — Breakdown by Provider
# -------------------------------------------------------
with tab5:
    st.subheader("LLM Breakdown by Provider")

    filenames = sorted(filtered["filename"].unique())
    selected_file = st.selectbox("Select Report Filename", filenames, key="file_tab5")

    pages = sorted(filtered[filtered["filename"] == selected_file]["page_number"].unique())
    selected_page = st.selectbox("Select Page Number", pages, key="page_tab5")

    subset = filtered[
        (filtered["filename"] == selected_file) &
        (filtered["page_number"] == selected_page)
    ].copy()

    providers = sorted(subset["provider"].unique())
    selected_provider = st.selectbox("Select Provider", providers, key="provider_tab5")

    provider_subset = subset[subset["provider"] == selected_provider]

    st.write("Models under provider:", sorted(provider_subset["model"].unique()))

    # --- COMPLETENESS FOR PROVIDER ---
    df_pdf = filtered[filtered["filename"] == selected_file]
    comp = model_completeness(df_pdf[df_pdf["provider"] == selected_provider], provider_subset)

    st.metric("Provider Completeness", f"{comp['score']*100:.1f}%")
    if comp["missing_count"] > 0:
        st.warning(
            f"Missing {comp['missing_count']} provider models: {', '.join(comp['missing'])}"
        )

    # Cleaned Markdown
    st.subheader("📖 Cleaned Markdown")
    if "cleaned_markdown" in subset.columns:
        st.markdown(subset["cleaned_markdown"].dropna().iloc[0])

    # Sentence comparison
    pivot_sent = provider_subset.pivot_table(
        index="sentence",
        columns="model",
        values="sentiment",
        aggfunc=lambda x: x.iloc[0] if len(x) else None
    )
    pivot_sent = ensure_all_models(df_pdf[df_pdf["provider"] == selected_provider], pivot_sent)
    st.dataframe(pivot_sent, use_container_width=True)

# -------------------------------------------------------
# TAB 6 — Model Coverage
# -------------------------------------------------------
with tab6:
    st.subheader("📦 Model Coverage Across PDFs and Pages")

    models_per_pdf = (
        df.groupby("filename")["model"].nunique()
        .rename("unique_model_count")
        .sort_values(ascending=False)
    )
    st.dataframe(models_per_pdf)

    models_per_page = (
        df.groupby(["filename", "page_number"])["model"]
        .nunique()
        .reset_index()
        .rename(columns={"model": "unique_model_count"})
    )
    st.dataframe(models_per_page)

    selected_file_cov = st.selectbox(
        "Select Report Filename",
        sorted(df["filename"].unique()),
        key="cov_file"
    )

    st.subheader("📄 Pages for this File")
    subset = models_per_page[
        models_per_page["filename"] == selected_file_cov
    ].sort_values("page_number")
    st.dataframe(subset)

    st.subheader("🧠 Models Used on Each Page")
    model_page_map = (
        df[df["filename"] == selected_file_cov]
        .groupby("page_number")["model"]
        .unique()
        .reset_index()
    )
    model_page_map["models"] = model_page_map["model"].apply(lambda x: ", ".join(sorted(x)))
    model_page_map = model_page_map.drop(columns=["model"])
    st.dataframe(model_page_map)

    st.subheader("🔥 Model–Page Heatmap")
    pivot = (
        df[df["filename"] == selected_file_cov]
        .pivot_table(
            index="page_number",
            columns="model",
            values="sentence",
            aggfunc="count",
            fill_value=0
        )
    )
    pivot = ensure_all_models(df[df["filename"] == selected_file_cov], pivot)
    st.dataframe(pivot.style.background_gradient(cmap="Blues"), use_container_width=True)

# -------------------------------------------------------
# TAB 7 — Raw JSON View (FIXED)
# -------------------------------------------------------
with tab7:
    st.subheader("📦 Raw JSON Data Viewer")

    filenames = sorted(raw_df["filename"].unique())
    selected_file = st.selectbox("Filename", filenames, key="raw_file")

    pages = sorted(raw_df[raw_df["filename"] == selected_file]["page_number"].unique())
    selected_page = st.selectbox("Page", pages, key="raw_page")

    subset = raw_df[
        (raw_df["filename"] == selected_file) &
        (raw_df["page_number"] == selected_page)
    ]

    for _, row in subset.iterrows():
        st.markdown(f"## 🤖 Model: **{row['model']}**")

        with st.expander("📄 Raw Text"):
            st.code(row["text"], language="json")

        parsed = parse_esg_json(row["text"])
        st.caption(f"Parsed {len(parsed)} ESG objects")

        with st.expander("✅ Parsed JSON"):
            st.json(parsed)

        if parsed:
            with st.expander("📊 Normalized Table"):
                st.dataframe(pd.json_normalize(parsed), use_container_width=True)


# import streamlit as st
# import pandas as pd
# import json
# import os

# # -------------------------------------------------------
# # Page Config
# # -------------------------------------------------------
# st.set_page_config(page_title="Parsed ESG JSON Dashboard", layout="wide")
# st.title("📊 ESG Parsed Sentence-Level Dashboard")

# # -------------------------------------------------------
# # Load CSV
# # -------------------------------------------------------
# DATA_PATH = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "..", "..", "data", "data_output.csv")
# )

# st.caption(f"Using data: `{DATA_PATH}`")

# @st.cache_data
# def load_data(path):
#     return pd.read_csv(path)

# try:
#     raw_df = load_data(DATA_PATH)
# except Exception as e:
#     st.error(f"❌ Failed to load CSV at: {DATA_PATH}\n\n{e}")
#     st.stop()

# # -------------------------------------------------------
# # Parse JSON annotations
# # -------------------------------------------------------
# def parse_annotations(df):
#     df = df.copy()

#     def safe_load(x):
#         try:
#             return json.loads(x)
#         except:
#             return []

#     df["parsed"] = df["text"].apply(safe_load)

#     exploded = df.explode("parsed", ignore_index=True)
#     parsed_df = pd.json_normalize(exploded["parsed"])

#     meta_cols = [c for c in df.columns if c != "parsed"]
#     meta = exploded[meta_cols].reset_index(drop=True)

#     full = pd.concat([meta, parsed_df], axis=1)
#     return full

# df = parse_annotations(raw_df)

# # -------------------------------------------------------
# # Helper: Parse provider
# # -------------------------------------------------------
# def parse_provider(m):
#     if isinstance(m, str) and "/" in m:
#         return m.split("/")[0]
#     return "unknown"

# df["provider"] = df["model"].apply(parse_provider)

# # -------------------------------------------------------
# # Helper: Ensure pivot contains ALL models (0 if missing)
# # -------------------------------------------------------
# def ensure_all_models(reference_df, pivot):
#     all_models = sorted(reference_df["model"].dropna().unique())
#     for m in all_models:
#         if m not in pivot.columns:
#             pivot[m] = 0
#     return pivot[all_models]

# # -------------------------------------------------------
# # Helper: completeness scoring
# # -------------------------------------------------------
# def model_completeness(df_pdf, df_page):
#     expected = sorted(df_pdf["model"].dropna().unique())
#     present = sorted(df_page["model"].dropna().unique())
#     missing = set(expected) - set(present)

#     score = len(present) / len(expected) if expected else 1.0

#     return {
#         "expected": expected,
#         "present": present,
#         "missing": sorted(missing),
#         "missing_count": len(missing),
#         "present_count": len(present),
#         "total": len(expected),
#         "score": score
#     }

# # -------------------------------------------------------
# # Sidebar Filters
# # -------------------------------------------------------
# st.sidebar.header("🔍 Filters")

# def make_multiselect(label, col):
#     if col not in df.columns:
#         return None
#     vals = sorted(df[col].dropna().unique())
#     return st.sidebar.multiselect(label, vals, default=vals)

# aspect_cats = make_multiselect("Aspect Category", "aspect_category")
# sentiments = make_multiselect("Sentiment", "sentiment")
# tones = make_multiselect("Tone", "tone")
# materialities = make_multiselect("Materiality", "materiality")
# stakeholders = make_multiselect("Stakeholder", "stakeholder")
# value_chain_stage = make_multiselect("Value Chain Stage", "value_chain_stage")
# time_horizon = make_multiselect("Time Horizon", "time_horizon")

# if "confidence" in df.columns:
#     conf_range = st.sidebar.slider(
#         "Confidence Range",
#         0.0, 1.0,
#         (float(df["confidence"].min()), float(df["confidence"].max())),
#         0.01,
#     )
# else:
#     conf_range = None

# # Apply filters
# filtered = df.copy()

# def apply_filter(col, values):
#     global filtered
#     if values and col in filtered.columns:
#         filtered = filtered[filtered[col].isin(values)]

# apply_filter("aspect_category", aspect_cats)
# apply_filter("sentiment", sentiments)
# apply_filter("tone", tones)
# apply_filter("materiality", materialities)
# apply_filter("stakeholder", stakeholders)
# apply_filter("value_chain_stage", value_chain_stage)
# apply_filter("time_horizon", time_horizon)

# if conf_range:
#     lo, hi = conf_range
#     filtered = filtered[(filtered["confidence"] >= lo) & (filtered["confidence"] <= hi)]

# st.caption(f"Showing **{len(filtered)}** sentences after filtering.")

# # -------------------------------------------------------
# # Tabs
# # -------------------------------------------------------
# tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
#     "📊 Distributions",
#     "📌 Aspects",
#     "📄 Sentence Table",
#     "🤖 LLM Model Comparison",
#     "LLM Breakdown",
#     "🧮 Model Coverage",
#     "📦 Raw JSON View"
# ])

# # -------------------------------------------------------
# # TAB 1 — Distributions
# # -------------------------------------------------------
# with tab1:
#     st.subheader("Sentiment Distribution")
#     st.bar_chart(filtered["sentiment"].value_counts())

#     st.subheader("Aspect Category Distribution")
#     st.bar_chart(filtered["aspect_category"].value_counts())

# # -------------------------------------------------------
# # TAB 2 — Aspects
# # -------------------------------------------------------
# with tab2:
#     st.subheader("Top Aspects")
#     if "aspect" in filtered:
#         n = st.slider("Show Top N", 3, 30, 10)
#         topA = filtered["aspect"].value_counts().head(n)
#         st.bar_chart(topA)
#         st.dataframe(topA.rename("count"))

# # -------------------------------------------------------
# # TAB 3 — Sentence Table
# # -------------------------------------------------------
# with tab3:
#     st.subheader("Full Sentence Table")
#     wanted = [
#         "sentence","aspect","aspect_category","sentiment","sentiment_score",
#         "tone","materiality","stakeholder","impact_level","time_horizon",
#         "filename","page_number"
#     ]
#     show_cols = [c for c in wanted if c in filtered.columns]
#     st.dataframe(filtered[show_cols], use_container_width=True)

# # -------------------------------------------------------
# # TAB 4 — Model Comparison (All Models)
# # -------------------------------------------------------
# with tab4:
#     st.subheader("LLM Model Comparison for Same File & Page")

#     filenames = sorted(filtered["filename"].unique())
#     selected_file = st.selectbox("Select Report Filename", filenames, key="file_tab4")

#     pages = sorted(filtered[filtered["filename"] == selected_file]["page_number"].unique())
#     selected_page = st.selectbox("Select Page Number", pages, key="page_tab4")

#     subset = filtered[
#         (filtered["filename"] == selected_file) &
#         (filtered["page_number"] == selected_page)
#     ]

#     # ----- COMPLETENESS SCORE -----
#     df_pdf = filtered[filtered["filename"] == selected_file]
#     comp = model_completeness(df_pdf, subset)

#     st.metric("Model Completeness", f"{comp['score']*100:.1f}%")
#     st.caption(f"{comp['present_count']} of {comp['total']} models appear on this page.")

#     if comp["score"] < 0.7:
#         st.warning(
#             f"⚠️ Only {comp['present_count']} of {comp['total']} expected models "
#             f"produced annotations.\nMissing models: {', '.join(comp['missing'])}"
#         )

#     st.markdown(f"### Comparing `{selected_file}` — Page **{selected_page}**")

#     # ----- SENTENCE COMPARISON -----
#     comparison = subset.pivot_table(
#         index="sentence",
#         columns="model",
#         values="sentiment",
#         aggfunc=lambda x: x.iloc[0] if len(x) else None,
#         dropna=False
#     )
#     comparison = ensure_all_models(df_pdf, comparison)
#     st.subheader("🔍 Sentence-Level Comparison")
#     st.dataframe(comparison, use_container_width=True)

#     # ----- ASPECT COMPARISON -----
#     aspect = subset.pivot_table(
#         index="sentence",
#         columns="model",
#         values="aspect",
#         aggfunc=lambda x: x.iloc[0] if len(x) else None,
#         dropna=False
#     )
#     aspect = ensure_all_models(df_pdf, aspect)
#     st.subheader("📌 Aspect Differences")
#     st.dataframe(aspect, use_container_width=True)

# # -------------------------------------------------------
# # TAB 5 — Breakdown by Provider
# # -------------------------------------------------------
# with tab5:
#     st.subheader("LLM Breakdown by Provider")

#     filenames = sorted(filtered["filename"].unique())
#     selected_file = st.selectbox("Select Report Filename", filenames, key="file_tab5")

#     pages = sorted(filtered[filtered["filename"] == selected_file]["page_number"].unique())
#     selected_page = st.selectbox("Select Page Number", pages, key="page_tab5")

#     subset = filtered[
#         (filtered["filename"] == selected_file) &
#         (filtered["page_number"] == selected_page)
#     ].copy()

#     providers = sorted(subset["provider"].unique())
#     selected_provider = st.selectbox("Select Provider", providers, key="provider_tab5")

#     provider_subset = subset[subset["provider"] == selected_provider]

#     st.write("Models under provider:", sorted(provider_subset["model"].unique()))

#     # --- COMPLETENESS FOR PROVIDER ---
#     df_pdf = filtered[filtered["filename"] == selected_file]
#     comp = model_completeness(df_pdf[df_pdf["provider"] == selected_provider], provider_subset)

#     st.metric("Provider Completeness", f"{comp['score']*100:.1f}%")
#     if comp["missing_count"] > 0:
#         st.warning(
#             f"Missing {comp['missing_count']} provider models: {', '.join(comp['missing'])}"
#         )

#     # Cleaned Markdown
#     st.subheader("📖 Cleaned Markdown")
#     if "cleaned_markdown" in subset.columns:
#         st.markdown(subset["cleaned_markdown"].dropna().iloc[0])

#     # Sentence comparison
#     pivot_sent = provider_subset.pivot_table(
#         index="sentence",
#         columns="model",
#         values="sentiment",
#         aggfunc=lambda x: x.iloc[0] if len(x) else None
#     )
#     pivot_sent = ensure_all_models(df_pdf[df_pdf["provider"] == selected_provider], pivot_sent)
#     st.dataframe(pivot_sent, use_container_width=True)

# # -------------------------------------------------------
# # TAB 6 — Model Coverage
# # -------------------------------------------------------
# with tab6:
#     st.subheader("📦 Model Coverage Across PDFs and Pages")

#     models_per_pdf = (
#         df.groupby("filename")["model"].nunique()
#         .rename("unique_model_count")
#         .sort_values(ascending=False)
#     )
#     st.dataframe(models_per_pdf)

#     models_per_page = (
#         df.groupby(["filename", "page_number"])["model"]
#         .nunique()
#         .reset_index()
#         .rename(columns={"model": "unique_model_count"})
#     )
#     st.dataframe(models_per_page)

#     selected_file_cov = st.selectbox(
#         "Select Report Filename",
#         sorted(df["filename"].unique()),
#         key="cov_file"
#     )

#     st.subheader("📄 Pages for this File")
#     subset = models_per_page[
#         models_per_page["filename"] == selected_file_cov
#     ].sort_values("page_number")
#     st.dataframe(subset)

#     st.subheader("🧠 Models Used on Each Page")
#     model_page_map = (
#         df[df["filename"] == selected_file_cov]
#         .groupby("page_number")["model"]
#         .unique()
#         .reset_index()
#     )
#     model_page_map["models"] = model_page_map["model"].apply(lambda x: ", ".join(sorted(x)))
#     model_page_map = model_page_map.drop(columns=["model"])
#     st.dataframe(model_page_map)

#     st.subheader("🔥 Model–Page Heatmap")
#     pivot = (
#         df[df["filename"] == selected_file_cov]
#         .pivot_table(
#             index="page_number",
#             columns="model",
#             values="sentence",
#             aggfunc="count",
#             fill_value=0
#         )
#     )
#     pivot = ensure_all_models(df[df["filename"] == selected_file_cov], pivot)
#     st.dataframe(pivot.style.background_gradient(cmap="Blues"), use_container_width=True)


# with tab7:
#     st.subheader("📦 Raw JSON Data Viewer")

#     # Step 1: Select file + page
#     filenames = sorted(df["filename"].dropna().unique())
#     selected_file_raw = st.selectbox(
#         "Select Report Filename",
#         filenames,
#         key="raw_file_tab"
#     )

#     pages = sorted(df[df["filename"] == selected_file_raw]["page_number"].dropna().unique())
#     selected_page_raw = st.selectbox(
#         "Select Page Number",
#         pages,
#         key="raw_page_tab"
#     )

#     # Subset relevant rows
#     raw_subset = raw_df[
#         (raw_df["filename"] == selected_file_raw) &
#         (raw_df["page_number"] == selected_page_raw)
#     ]

#     st.markdown(
#         f"### Raw JSON Entries for `{selected_file_raw}` — Page **{selected_page_raw}**"
#     )

#     if raw_subset.empty:
#         st.warning("⚠️ No raw JSON data found for the selected page.")
#         st.stop()

#     st.info(f"Found **{len(raw_subset)}** JSON entries on this page.")

#     # Iterate through each row (each model output)
#     for idx, row in raw_subset.iterrows():
#         model_name = row.get("model", "unknown-model")
#         raw_json_text = row.get("text", "[]")

#         st.markdown(f"## 🤖 Model: **{model_name}**")

#         # Collapsible raw JSON
#         with st.expander("🔎 Show Raw JSON String (text field)"):
#             st.code(raw_json_text, language="json")

#         # Try parsing JSON
#         try:
#             parsed_json = json.loads(raw_json_text)
#         except Exception as e:
#             st.error(f"❌ JSON parsing error: {e}")
#             continue

#         # Parsed JSON viewer
#         with st.expander("📄 Parsed JSON Objects"):
#             st.json(parsed_json)

#         # Normalized into DataFrame
#         parsed_df = pd.json_normalize(parsed_json)

#         with st.expander("📊 Normalized Table View"):
#             st.dataframe(parsed_df, use_container_width=True)


# import streamlit as st
# import pandas as pd
# import json
# import os

# # Don't call set_page_config here if you already call it in dashboard/app.py.
# # If you get a warning about multiple set_page_config calls, you can safely delete this line.
# st.set_page_config(page_title="Parsed ESG JSON Dashboard", layout="wide")

# st.title("📊 ESG Parsed Sentence-Level Dashboard")

# # -------------------------------------------------------
# # Load CSV Automatically
# # -------------------------------------------------------

# # __file__ = .../esg_dashboard_new/dashboard/pages/8_new.py
# # We want:                            ../../data/data_output.csv
# DATA_PATH = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "..", "..", "data", "data_output.csv")
# )

# st.caption(f"Using data file: `{DATA_PATH}`")

# @st.cache_data
# def load_data(path: str) -> pd.DataFrame:
#     return pd.read_csv(path)

# try:
#     raw_df = load_data(DATA_PATH)
# except Exception as e:
#     st.error(f"❌ Failed to load CSV at:\n`{DATA_PATH}`\n\nError: {e}")
#     st.stop()


# # -------------------------------------------------------
# # Parse JSON list stored in column: text
# # -------------------------------------------------------
# def parse_annotations(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()

#     if "text" not in df.columns:
#         st.error("❌ Column 'text' missing in CSV. Available columns: "
#                  + ", ".join(df.columns.astype(str)))
#         st.stop()

#     def safe_json_load(x):
#         try:
#             return json.loads(x)
#         except Exception:
#             return []

#     df["parsed"] = df["text"].apply(safe_json_load)

#     exploded = df.explode("parsed", ignore_index=True)

#     parsed_df = pd.json_normalize(exploded["parsed"])

#     meta_cols = [c for c in df.columns if c not in ["parsed"]]
#     meta = exploded[meta_cols].reset_index(drop=True)

#     full = pd.concat([meta, parsed_df], axis=1)
#     return full


# df = parse_annotations(raw_df)

# # -------------------------------------------------------
# # Sidebar Filters
# # -------------------------------------------------------
# st.sidebar.header("🔍 Filters")

# def make_multiselect(label: str, column: str):
#     if column not in df.columns:
#         return None
#     values = sorted(df[column].dropna().unique())
#     return st.sidebar.multiselect(label, values, default=values)

# aspect_cats = make_multiselect("Aspect Category", "aspect_category")
# sentiments = make_multiselect("Sentiment", "sentiment")
# tones = make_multiselect("Tone", "tone")
# materialities = make_multiselect("Materiality", "materiality")
# stakeholders = make_multiselect("Stakeholder", "stakeholder")
# value_chain_stage = make_multiselect("Value Chain Stage", "value_chain_stage")
# time_horizon = make_multiselect("Time Horizon", "time_horizon")

# # Confidence filter
# if "confidence" in df.columns:
#     min_conf, max_conf = float(df["confidence"].min()), float(df["confidence"].max())
#     conf_range = st.sidebar.slider(
#         "Confidence Range",
#         0.0,
#         1.0,
#         (min_conf, max_conf),
#         0.01
#     )
# else:
#     conf_range = None

# # -------------------------------------------------------
# # Apply Filters
# # -------------------------------------------------------
# filtered = df.copy()

# def apply_filter(col: str, selected_values):
#     global filtered
#     if selected_values is not None and col in filtered.columns:
#         filtered = filtered[filtered[col].isin(selected_values)]

# apply_filter("aspect_category", aspect_cats)
# apply_filter("sentiment", sentiments)
# apply_filter("tone", tones)
# apply_filter("materiality", materialities)
# apply_filter("stakeholder", stakeholders)
# apply_filter("value_chain_stage", value_chain_stage)
# apply_filter("time_horizon", time_horizon)

# if conf_range and "confidence" in filtered.columns:
#     low, high = conf_range
#     filtered = filtered[
#         (filtered["confidence"] >= low) & (filtered["confidence"] <= high)
#     ]

# st.caption(f"Showing **{len(filtered)}** sentences after filtering.")

# # -------------------------------------------------------
# # Summary Metrics
# # -------------------------------------------------------
# col1, col2, col3, col4 = st.columns(4)

# with col1:
#     st.metric("Total Sentences", len(filtered))

# with col2:
#     if "sentiment" in filtered.columns and len(filtered) > 0:
#         values = filtered["sentiment"].value_counts(normalize=True) * 100
#         pos = values.get("positive", 0)
#         neu = values.get("neutral", 0)
#         neg = values.get("negative", 0)
#         st.metric(
#             "Positive / Neutral / Negative (%)",
#             f"{pos:.1f} / {neu:.1f} / {neg:.1f}"
#         )
#     else:
#         st.metric("Sentiment", "-")

# with col3:
#     if "confidence" in filtered.columns and len(filtered) > 0:
#         st.metric("Avg. Model Confidence", f"{filtered['confidence'].mean():.2f}")
#     else:
#         st.metric("Avg. Confidence", "-")

# with col4:
#     if "materiality" in filtered.columns and len(filtered) > 0:
#         top_mat = filtered["materiality"].value_counts().idxmax()
#         st.metric("Most Common Materiality", top_mat)
#     else:
#         st.metric("Materiality", "-")


# # -------------------------------------------------------
# # Tabs: Charts / Aspects / Full Table
# # -------------------------------------------------------
# tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Distributions", "📌 Aspects", "📄 Sentence Table",  "🤖 LLM Model Comparison", "LLM Breakdown", "🧮 Model Coverage"])

# with tab1:
#     st.subheader("Sentiment Distribution")
#     if "sentiment" in filtered.columns and len(filtered) > 0:
#         st.bar_chart(filtered["sentiment"].value_counts())

#     st.subheader("Aspect Category Distribution")
#     if "aspect_category" in filtered.columns and len(filtered) > 0:
#         st.bar_chart(filtered["aspect_category"].value_counts())

#     if {"materiality", "impact_level"}.issubset(filtered.columns) and len(filtered) > 0:
#         st.subheader("Materiality × Impact Level Table")
#         st.dataframe(pd.crosstab(filtered["materiality"], filtered["impact_level"]))


# with tab2:
#     st.subheader("Top Aspects")
#     if "aspect" in filtered.columns and len(filtered) > 0:
#         n = st.slider("Show Top N", 3, 30, 10)
#         aspect_counts = filtered["aspect"].value_counts().head(n)
#         st.bar_chart(aspect_counts)
#         st.dataframe(aspect_counts.rename("count"))


# with tab3:
#     st.subheader("Full Sentence Table")
#     main_cols = [
#         "sentence",
#         "aspect",
#         "aspect_category",
#         "sentiment",
#         "sentiment_score",
#         "tone",
#         "materiality",
#         "stakeholder",
#         "impact_level",
#         "time_horizon",
#         "filename",
#         "page_number",
#     ]
#     show_cols = [c for c in main_cols if c in filtered.columns]

#     st.dataframe(filtered[show_cols])

#     if "markdown_full" in filtered.columns:
#         with st.expander("📖 Show Document Context (markdown_full)"):
#             for fname, group in filtered.groupby("filename"):
#                 st.markdown(f"### 📄 {fname}")
#                 ctx = group["markdown_full"].dropna().iloc[0]
#                 st.markdown(ctx)

# st.success("Dashboard loaded successfully ✔️")

# # ------------------------------------------
# # Define parse_provider BEFORE tabs
# # ------------------------------------------
# def parse_provider(m):
#     if isinstance(m, str) and "/" in m:
#         return m.split("/")[0]
#     return "unknown"


# # ------------------------------------------
# # LLM Model Comparison (Tab 4) — ORIGINAL VERSION
# # ------------------------------------------
# with tab4:
#     st.subheader("LLM Model Comparison for Same File & Page")

#     # Step 1: Select file + page
#     filenames = sorted(filtered["filename"].dropna().unique())
#     selected_file = st.selectbox("Select Report Filename", filenames, key="file_tab4")

#     pages = sorted(filtered[filtered["filename"] == selected_file]["page_number"].dropna().unique())
#     selected_page = st.selectbox("Select Page Number", pages, key="page_tab4")

#     subset = filtered[
#         (filtered["filename"] == selected_file) &
#         (filtered["page_number"] == selected_page)
#     ]

#     st.markdown(f"### 📄 Comparing models for `{selected_file}` — Page **{selected_page}**")

#     # Step 2: Show models present
#     models_available = sorted(subset["model"].dropna().unique())
#     st.write("Models detected:", models_available)

#     # Step 3: Sentence Level Comparison
#     st.subheader("🔍 Sentence-Level Comparison Across Models")

#     comparison_table = subset.pivot_table(
#         index="sentence",
#         columns="model",
#         values="sentiment",
#         aggfunc=lambda x: x.iloc[0]
#     )
#     st.dataframe(comparison_table, use_container_width=True)

#     # Aspect comparison
#     st.subheader("📌 Aspect Label Differences")
#     aspect_table = subset.pivot_table(
#         index="sentence",
#         columns="model",
#         values="aspect",
#         aggfunc=lambda x: x.iloc[0]
#     )
#     st.dataframe(aspect_table, use_container_width=True)

#     # Sentiment distribution
#     st.subheader("📊 Sentiment Distribution by Model")
#     sent_counts = subset.groupby("model")["sentiment"].value_counts().unstack(fill_value=0)
#     st.bar_chart(sent_counts)

#     # Aspect category distribution
#     st.subheader("🏷️ Aspect-Category Distribution")
#     ac_counts = subset.groupby("model")["aspect_category"].value_counts().unstack(fill_value=0)
#     st.bar_chart(ac_counts)

#     # Agreement
#     st.subheader("📈 Model Agreement Metrics")

#     def sentiment_agreement(df):
#         pivot = df.pivot_table(index="sentence", columns="model", values="sentiment", aggfunc=lambda x: x.iloc[0])
#         return (pivot.nunique(axis=1) == 1).mean()

#     def aspect_agreement(df):
#         pivot = df.pivot_table(index="sentence", columns="model", values="aspect", aggfunc=lambda x: x.iloc[0])
#         return (pivot.nunique(axis=1) == 1).mean()

#     st.metric("Sentiment Agreement Rate", f"{sentiment_agreement(subset)*100:.1f}%")
#     st.metric("Aspect Agreement Rate", f"{aspect_agreement(subset)*100:.1f}%")


# # ------------------------------------------
# # LLM Breakdown (Tab 5) — PROVIDER VERSION
# # ------------------------------------------
# with tab5:
#     st.subheader("LLM Breakdown by Provider")

#     # Step 1: Select file + page
#     filenames = sorted(filtered["filename"].dropna().unique())
#     selected_file = st.selectbox("Select Report Filename", filenames, key="file_tab5")

#     pages = sorted(filtered[filtered["filename"] == selected_file]["page_number"].dropna().unique())
#     selected_page = st.selectbox("Select Page Number", pages, key="page_tab5")

#     subset = filtered[
#         (filtered["filename"] == selected_file) &
#         (filtered["page_number"] == selected_page)
#     ].copy()

#     # Assign provider column
#     subset["provider"] = subset["model"].apply(parse_provider)

#     providers = sorted(subset["provider"].unique())
#     selected_provider = st.selectbox("Select Provider", providers, key="provider_tab5")

#     provider_subset = subset[subset["provider"] == selected_provider]

#     st.write("Models under this provider:", sorted(provider_subset["model"].unique()))

#     # Cleaned Markdown
#     st.subheader("📖 Cleaned Markdown")
#     if "cleaned_markdown" in provider_subset.columns:
#         st.markdown(provider_subset["cleaned_markdown"].dropna().iloc[0])

#     # Sentence comparison
#     st.subheader("📝 Sentence-Level Comparison")
#     pivot_sent = provider_subset.pivot_table(
#         index="sentence",
#         columns="model",
#         values="sentiment",
#         aggfunc=lambda x: x.iloc[0]
#     )
#     st.dataframe(pivot_sent, use_container_width=True)

#     # Aspect comparison
#     st.subheader("🏷️ Aspect Comparison")
#     pivot_aspect = provider_subset.pivot_table(
#         index="sentence",
#         columns="model",
#         values="aspect",
#         aggfunc=lambda x: x.iloc[0]
#     )
#     st.dataframe(pivot_aspect, use_container_width=True)

#     # Model distributions
#     st.subheader("📊 Sentiment Distribution by Model")
#     st.bar_chart(
#         provider_subset.groupby("model")["sentiment"]
#         .value_counts()
#         .unstack(fill_value=0)
#     )

#     st.subheader("📦 Aspect Category Distribution by Model")
#     st.bar_chart(
#         provider_subset.groupby("model")["aspect_category"]
#         .value_counts()
#         .unstack(fill_value=0)
#     )

# # tab6 = st.tabs(["🧮 Model Coverage"])[0]

# with tab6:
#     st.subheader("📦 Model Coverage Across PDFs and Pages")

#     # Count models per filename
#     st.markdown("### 📘 Model Count per PDF")

#     models_per_pdf = (
#         df.groupby("filename")["model"]
#         .nunique()
#         .sort_values(ascending=False)
#     ).rename("unique_model_count")

#     st.dataframe(models_per_pdf)

#     # Count models per page within each PDF
#     st.markdown("### 📄 Model Count per Page")

#     models_per_page = (
#         df.groupby(["filename", "page_number"])["model"]
#         .nunique()
#         .reset_index()
#         .rename(columns={"model": "unique_model_count"})
#     )

#     st.dataframe(models_per_page)

#     # Select filename to inspect pages
#     st.markdown("### 🔍 Inspect Model Coverage by Page")

#     filenames = sorted(df["filename"].dropna().unique())
#     selected_file_cov = st.selectbox(
#         "Select Report Filename",
#         filenames,
#         key="model_coverage_file"
#     )

#     subset = models_per_page[
#         models_per_page["filename"] == selected_file_cov
#     ].sort_values("page_number")

#     st.dataframe(subset)

#     # Which models appear on each page?
#     st.markdown("### 🧠 Models Used on Each Page")

#     model_page_map = (
#         df[df["filename"] == selected_file_cov]
#         .groupby("page_number")["model"]
#         .unique()
#         .reset_index()
#     )

#     # Format list for display
#     model_page_map["models"] = model_page_map["model"].apply(lambda x: ", ".join(sorted(x)))
#     model_page_map = model_page_map.drop(columns=["model"])

#     st.dataframe(model_page_map, use_container_width=True)

#     # Optional heatmap
#     st.markdown("### 🔥 Model–Page Heatmap")

#     pivot = (
#         df[df["filename"] == selected_file_cov]
#         .pivot_table(
#             index="page_number",
#             columns="model",
#             values="sentence",
#             aggfunc="count",
#             fill_value=0
#         )
#     )

#     st.dataframe(pivot.style.background_gradient(cmap="Blues"), use_container_width=True)


# ------------------------------------------
# LLM Model Comparison (Tab 4)
# ------------------------------------------
# with tab4:
#     st.subheader("LLM Model Comparison for Same File & Page")

#     # Step 1: Select file + page
#     filenames = sorted(df["filename"].dropna().unique())
#     selected_file = st.selectbox("Select Report Filename", filenames)

#     pages = sorted(df[df["filename"] == selected_file]["page_number"].dropna().unique())
#     selected_page = st.selectbox("Select Page Number", pages)

#     subset = df[(df["filename"] == selected_file) & (df["page_number"] == selected_page)]

#     st.markdown(f"### 📄 Comparing models for `{selected_file}` — Page **{selected_page}**")

#     # Step 2: Show models present
#     models_available = sorted(subset["model"].dropna().unique())
#     st.write("Models detected:", models_available)

#     # Step 3: Expand sentence-level comparison
#     st.subheader("🔍 Sentence-Level Comparison Across Models")

#     # Wide comparison table
#     comparison_table = subset.pivot_table(
#         index="sentence",
#         columns="model",
#         values="sentiment",
#         aggfunc=lambda x: x.iloc[0] if len(x) > 0 else None
#     )

#     st.dataframe(comparison_table, use_container_width=True)

#     # Step 4: Aspect comparison
#     st.subheader("📌 Aspect Label Differences")

#     aspect_table = subset.pivot_table(
#         index="sentence",
#         columns="model",
#         values="aspect",
#         aggfunc=lambda x: x.iloc[0] if len(x) > 0 else None
#     )
#     st.dataframe(aspect_table, use_container_width=True)

#     # Step 5: Sentiment distribution per model
#     st.subheader("📊 Sentiment Distribution by Model")

#     sent_counts = (
#         subset.groupby("model")["sentiment"]
#         .value_counts()
#         .unstack(fill_value=0)
#     )
#     st.bar_chart(sent_counts)

#     # Step 6: Aspect category distribution per model
#     st.subheader("🏷️ Aspect-Category Distribution")

#     ac_counts = (
#         subset.groupby("model")["aspect_category"]
#         .value_counts()
#         .unstack(fill_value=0)
#     )
#     st.bar_chart(ac_counts)

#     # Step 7: Compute agreement rate
#     st.subheader("📈 Model Agreement Metrics")

#     def sentiment_agreement(df):
#         pivot = df.pivot_table(
#             index="sentence",
#             columns="model",
#             values="sentiment",
#             aggfunc=lambda x: x.iloc[0]
#         )
#         # percent of rows where all models agree
#         return (pivot.nunique(axis=1) == 1).mean()

#     agreement_rate = sentiment_agreement(subset)
#     st.metric("Sentiment Agreement Rate", f"{agreement_rate*100:.1f}%")

#     # Aspect agreement
#     def aspect_agreement(df):
#         pivot = df.pivot_table(
#             index="sentence",
#             columns="model",
#             values="aspect",
#             aggfunc=lambda x: x.iloc[0]
#         )
#         return (pivot.nunique(axis=1) == 1).mean()

#     st.metric("Aspect Agreement Rate", f"{aspect_agreement(subset)*100:.1f}%")
# # with tab4:
# #     st.subheader("LLM Model Comparison for Same File & Page")

# #     # Step 1: Select File
# #     filenames = sorted(filtered["filename"].dropna().unique())
# #     selected_file = st.selectbox("Select Report Filename", filenames, key="file_tab4")

# #     pages = sorted(filtered[filtered["filename"] == selected_file]["page_number"].dropna().unique())
# #     selected_page = st.selectbox("Select Page Number", pages, key="page_tab4")

# #     subset = filtered[
# #         (filtered["filename"] == selected_file) &
# #         (filtered["page_number"] == selected_page)
# #     ].copy()

# #     st.markdown(f"### 📄 Comparing models for `{selected_file}` — Page **{selected_page}**")

# #     # Parse provider
# #     def parse_provider(m):
# #         if isinstance(m, str) and "/" in m:
# #             return m.split("/")[0]
# #         return "unknown"

# #     subset["provider"] = subset["model"].apply(parse_provider)

# #     providers = sorted(subset["provider"].unique())
# #     selected_provider = st.selectbox("Select Provider", providers, key="provider_tab4")

# #     provider_subset = subset[subset["provider"] == selected_provider].copy()

# #     st.write("Models under this provider:", sorted(provider_subset["model"].unique()))

# #     # Show cleaned_markdown
# #     st.subheader("📖 Page Context (cleaned_markdown)")
# #     if "cleaned_markdown" in provider_subset.columns:
# #         st.markdown(provider_subset["cleaned_markdown"].dropna().iloc[0])

# #     # Sentiment comparison
# #     st.subheader("📝 Sentence-Level Sentiment Comparison")
# #     pivot_sent = provider_subset.pivot_table(
# #         index="sentence",
# #         columns="model",
# #         values="sentiment",
# #         aggfunc=lambda x: x.iloc[0]
# #     )
# #     st.dataframe(pivot_sent, use_container_width=True)

# #     # Aspect comparison
# #     st.subheader("🏷️ Aspect Comparison")
# #     pivot_aspect = provider_subset.pivot_table(
# #         index="sentence",
# #         columns="model",
# #         values="aspect",
# #         aggfunc=lambda x: x.iloc[0]
# #     )
# #     st.dataframe(pivot_aspect, use_container_width=True)

# #     # Distributions
# #     st.subheader("📊 Sentiment Distribution by Model")
# #     st.bar_chart(
# #         provider_subset.groupby("model")["sentiment"]
# #         .value_counts()
# #         .unstack(fill_value=0)
# #     )

# # ------------------------------------------
# # LLM Breakdown (Tab 5)
# # ------------------------------------------
# with tab5:
#     st.subheader("LLM Breakdown by Provider")

#     filenames = sorted(filtered["filename"].dropna().unique())
#     selected_file = st.selectbox("Select Report Filename", filenames, key="file_tab5")

#     pages = sorted(filtered[filtered["filename"] == selected_file]["page_number"].dropna().unique())
#     selected_page = st.selectbox("Select Page Number", pages, key="page_tab5")

#     subset = filtered[
#         (filtered["filename"] == selected_file) &
#         (filtered["page_number"] == selected_page)
#     ].copy()

#     subset["provider"] = subset["model"].apply(parse_provider)

#     providers = sorted(subset["provider"].unique())
#     selected_provider = st.selectbox("Select Provider", providers, key="provider_tab5")

#     provider_subset = subset[subset["provider"] == selected_provider]

#     st.write("Models:", sorted(provider_subset["model"].unique()))

#     st.subheader("📖 Cleaned Markdown")
#     st.markdown(provider_subset["cleaned_markdown"].dropna().iloc[0])

#     st.subheader("Sentence Comparison")
#     pivot_sent = provider_subset.pivot_table(
#         index="sentence",
#         columns="model",
#         values="sentiment",
#         aggfunc=lambda x: x.iloc[0]
#     )
#     st.dataframe(pivot_sent, use_container_width=True)


# with tab4:
#     st.subheader("LLM Model Comparison for Same File & Page")

#     # Step 1: Select file + page
#     filenames = sorted(df["filename"].dropna().unique())
#     selected_file = st.selectbox("Select Report Filename", filenames)

#     pages = sorted(df[df["filename"] == selected_file]["page_number"].dropna().unique())
#     selected_page = st.selectbox("Select Page Number", pages)

#     subset = df[(df["filename"] == selected_file) & (df["page_number"] == selected_page)]

#     st.markdown(f"### 📄 Comparing models for `{selected_file}` — Page **{selected_page}**")

#     # Step 2: Show models present
#     models_available = sorted(subset["model"].dropna().unique())
#     st.write("Models detected:", models_available)

#     # Step 3: Expand sentence-level comparison
#     st.subheader("🔍 Sentence-Level Comparison Across Models")

#     # Wide comparison table
#     comparison_table = subset.pivot_table(
#         index="sentence",
#         columns="model",
#         values="sentiment",
#         aggfunc=lambda x: x.iloc[0] if len(x) > 0 else None
#     )

#     st.dataframe(comparison_table, use_container_width=True)

#     # Step 4: Aspect comparison
#     st.subheader("📌 Aspect Label Differences")

#     aspect_table = subset.pivot_table(
#         index="sentence",
#         columns="model",
#         values="aspect",
#         aggfunc=lambda x: x.iloc[0] if len(x) > 0 else None
#     )
#     st.dataframe(aspect_table, use_container_width=True)

#     # Step 5: Sentiment distribution per model
#     st.subheader("📊 Sentiment Distribution by Model")

#     sent_counts = (
#         subset.groupby("model")["sentiment"]
#         .value_counts()
#         .unstack(fill_value=0)
#     )
#     st.bar_chart(sent_counts)

#     # Step 6: Aspect category distribution per model
#     st.subheader("🏷️ Aspect-Category Distribution")

#     ac_counts = (
#         subset.groupby("model")["aspect_category"]
#         .value_counts()
#         .unstack(fill_value=0)
#     )
#     st.bar_chart(ac_counts)

#     # Step 7: Compute agreement rate
#     st.subheader("📈 Model Agreement Metrics")

#     def sentiment_agreement(df):
#         pivot = df.pivot_table(
#             index="sentence",
#             columns="model",
#             values="sentiment",
#             aggfunc=lambda x: x.iloc[0]
#         )
#         # percent of rows where all models agree
#         return (pivot.nunique(axis=1) == 1).mean()

#     agreement_rate = sentiment_agreement(subset)
#     st.metric("Sentiment Agreement Rate", f"{agreement_rate*100:.1f}%")

#     # Aspect agreement
#     def aspect_agreement(df):
#         pivot = df.pivot_table(
#             index="sentence",
#             columns="model",
#             values="aspect",
#             aggfunc=lambda x: x.iloc[0]
#         )
#         return (pivot.nunique(axis=1) == 1).mean()

#     st.metric("Aspect Agreement Rate", f"{aspect_agreement(subset)*100:.1f}%")

# with tab5:
#     st.subheader("LLM Model Comparison for Same File & Page")

#     # -------------------------------------------------------
#     # Step 1: Select File
#     # -------------------------------------------------------
#     filenames = sorted(df["filename"].dropna().unique())
#     selected_file = st.selectbox("Select Report Filename", filenames)

#     pages = sorted(df[df["filename"] == selected_file]["page_number"].dropna().unique())
#     selected_page = st.selectbox("Select Page Number", pages)

#     # Subset only this file + page
#     subset = df[
#         (df["filename"] == selected_file) &
#         (df["page_number"] == selected_page)
#     ].copy()

#     st.markdown(f"### 📄 Comparing models for `{selected_file}` — Page **{selected_page}**")

#     # -------------------------------------------------------
#     # Step 2: Extract LLM Providers
#     # -------------------------------------------------------
#     def parse_provider(m):
#         if isinstance(m, str) and "/" in m:
#             return m.split("/")[0].strip()
#         return "unknown"

#     subset["provider"] = subset["model"].apply(parse_provider)

#     providers = sorted(subset["provider"].unique())
#     selected_provider = st.selectbox("Select Provider", providers)

#     provider_subset = subset[subset["provider"] == selected_provider].copy()

#     st.markdown(f"### 🔍 Models available under `{selected_provider}`")
#     st.write(sorted(provider_subset["model"].unique()))

#     # -------------------------------------------------------
#     # Show cleaned_markdown
#     # -------------------------------------------------------
#     st.subheader("📖 Page Context (cleaned_markdown)")

#     if "cleaned_markdown" in provider_subset.columns:
#         try:
#             md = provider_subset["cleaned_markdown"].dropna().iloc[0]
#             st.markdown(md)
#         except:
#             st.info("No cleaned_markdown found for this selection.")
#     else:
#         st.info("Column 'cleaned_markdown' not found in dataset.")

#     # -------------------------------------------------------
#     # Sentence-level comparison across models
#     # -------------------------------------------------------
#     st.subheader("📝 Sentence-Level Sentiment Comparison")

#     if "sentence" in provider_subset.columns:
#         sentiment_matrix = provider_subset.pivot_table(
#             index="sentence",
#             columns="model",
#             values="sentiment",
#             aggfunc=lambda x: x.iloc[0] if len(x) > 0 else None
#         )
#         st.dataframe(sentiment_matrix, use_container_width=True)
#     else:
#         st.warning("No 'sentence' column available for comparison.")

#     # -------------------------------------------------------
#     # Aspect comparison
#     # -------------------------------------------------------
#     st.subheader("🏷️ Aspect Comparison")

#     if "aspect" in provider_subset.columns:
#         aspect_matrix = provider_subset.pivot_table(
#             index="sentence",
#             columns="model",
#             values="aspect",
#             aggfunc=lambda x: x.iloc[0] if len(x) > 0 else None
#         )
#         st.dataframe(aspect_matrix, use_container_width=True)
#     else:
#         st.warning("No 'aspect' column available.")

#     # -------------------------------------------------------
#     # Sentiment distribution per model
#     # -------------------------------------------------------
#     st.subheader("📊 Sentiment Distribution by Model")

#     sent_counts = (
#         provider_subset.groupby("model")["sentiment"]
#         .value_counts()
#         .unstack(fill_value=0)
#     )

#     st.bar_chart(sent_counts)

#     # -------------------------------------------------------
#     # Aspect category distribution
#     # -------------------------------------------------------
#     st.subheader("📦 Aspect Category Distribution by Model")

#     if "aspect_category" in provider_subset.columns:
#         ac_counts = (
#             provider_subset.groupby("model")["aspect_category"]
#             .value_counts()
#             .unstack(fill_value=0)
#         )
#         st.bar_chart(ac_counts)

#     # -------------------------------------------------------
#     # Agreement calculations
#     # -------------------------------------------------------
#     st.subheader("📈 Model Agreement Scores")

#     def agreement(df, column):
#         p = df.pivot_table(
#             index="sentence",
#             columns="model",
#             values=column,
#             aggfunc=lambda x: x.iloc[0] if len(x) > 0 else None
#         )
#         if p.empty:
#             return 0.0
#         return (p.nunique(axis=1) == 1).mean()

#     sentiment_agree = agreement(provider_subset, "sentiment")
#     aspect_agree = agreement(provider_subset, "aspect")

#     st.metric("Sentiment Agreement Rate", f"{sentiment_agree*100:.1f}%")
#     st.metric("Aspect Agreement Rate", f"{aspect_agree*100:.1f}%")


# import json
# import pandas as pd
# import streamlit as st

# st.set_page_config(
#     page_title="ESG Feedback & Sustainability Report Dashboard",
#     layout="wide"
# )

# st.title("📊 ESG Feedback & Sustainability Report Dashboard")

# st.markdown(
#     """
# This dashboard visualises ESG-style annotations (aspect, sentiment, tone, materiality, etc.)  
# extracted from Sustainability Reports.
# """
# )

# # -------------------------------------------------------
# # Data loading
# # -------------------------------------------------------
# @st.cache_data
# def load_data(uploaded_file: bytes) -> pd.DataFrame:
#     df = pd.read_csv(uploaded_file)
#     return df


# def parse_annotations(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     - Assumes df has a column 'text' that is a JSON list of sentence-level dicts.
#     - Explodes into one row per sentence with document metadata attached.
#     """
#     if "text" not in df.columns:
#         raise ValueError("Expected a 'text' column containing the JSON annotations.")

#     # Parse JSON list from text column
#     df = df.copy()
#     df["parsed"] = df["text"].apply(lambda x: json.loads(x) if pd.notna(x) else [])

#     # Explode into rows
#     exploded = df.explode("parsed", ignore_index=True)

#     # Normalise the dict into columns
#     parsed_df = pd.json_normalize(exploded["parsed"])

#     # Attach doc-level metadata
#     meta_cols = [c for c in df.columns if c not in ["parsed"]]
#     meta = exploded[meta_cols].reset_index(drop=True)

#     full = pd.concat([meta, parsed_df], axis=1)

#     # Optional: nicer column order if present
#     preferred_order = [
#         "sentence",
#         "aspect",
#         "aspect_category",
#         "sentiment",
#         "sentiment_score",
#         "tone",
#         "materiality",
#         "esg_risk_type",
#         "impact_level",
#         "stakeholder",
#         "time_horizon",
#         "claim_type",
#         "has_kpi",
#         "has_target",
#         "ontology_uri",
#         "regulation_uri",
#         "value_chain_stage",
#         "emission_scope",
#         "policy_reference",
#         "partner_type",
#         "confidence",
#         "page_number",
#         "filename",
#         "filename_index",
#     ]

#     cols = [c for c in preferred_order if c in full.columns] + [
#         c for c in full.columns if c not in preferred_order
#     ]
#     full = full[cols]

#     return full


# # -------------------------------------------------------
# # Sidebar: upload + filters
# # -------------------------------------------------------
# st.sidebar.header("⚙️ Settings")

# uploaded_file = st.sidebar.file_uploader(
#     "Upload CSV file with annotations",
#     type=["csv"],
#     help="CSV must contain a 'text' column with JSON list of sentence-level annotations."
# )

# if uploaded_file is None:
#     st.info("Upload a CSV file in the sidebar to start exploring the dashboard.")
#     st.stop()

# try:
#     raw_df = load_data(uploaded_file)
#     df = parse_annotations(raw_df)
# except Exception as e:
#     st.error(f"Error parsing file: {e}")
#     st.stop()

# st.sidebar.subheader("🔍 Filters")

# def multiselect_filter(label, col_name):
#     if col_name not in df.columns:
#         return None
#     options = sorted([x for x in df[col_name].dropna().unique()])
#     selected = st.sidebar.multiselect(label, options, default=options)
#     return selected

# aspect_cats = multiselect_filter("Aspect Category", "aspect_category")
# sentiments = multiselect_filter("Sentiment", "sentiment")
# tones = multiselect_filter("Tone", "tone")
# materialities = multiselect_filter("Materiality", "materiality")
# stakeholders = multiselect_filter("Stakeholder", "stakeholder")
# value_chain_stages = multiselect_filter("Value Chain Stage", "value_chain_stage")
# time_horizons = multiselect_filter("Time Horizon", "time_horizon")

# # Confidence filter
# if "confidence" in df.columns:
#     min_conf, max_conf = float(df["confidence"].min()), float(df["confidence"].max())
#     conf_range = st.sidebar.slider(
#         "Confidence range",
#         min_value=0.0,
#         max_value=1.0,
#         value=(min_conf, max_conf),
#         step=0.01,
#     )
# else:
#     conf_range = None

# # -------------------------------------------------------
# # Apply filters
# # -------------------------------------------------------
# filtered = df.copy()

# if aspect_cats is not None:
#     filtered = filtered[filtered["aspect_category"].isin(aspect_cats)]

# if sentiments is not None:
#     filtered = filtered[filtered["sentiment"].isin(sentiments)]

# if tones is not None:
#     filtered = filtered[filtered["tone"].isin(tones)]

# if materialities is not None:
#     filtered = filtered[filtered["materiality"].isin(materialities)]

# if stakeholders is not None:
#     filtered = filtered[filtered["stakeholder"].isin(stakeholders)]

# if value_chain_stages is not None:
#     filtered = filtered[filtered["value_chain_stage"].isin(value_chain_stages)]

# if time_horizons is not None:
#     filtered = filtered[filtered["time_horizon"].isin(time_horizons)]

# if conf_range is not None and "confidence" in filtered.columns:
#     low, high = conf_range
#     filtered = filtered[
#         (filtered["confidence"] >= low) & (filtered["confidence"] <= high)
#     ]

# st.caption(f"Showing **{len(filtered)}** annotated sentences.")


# # -------------------------------------------------------
# # Top summary metrics
# # -------------------------------------------------------
# col1, col2, col3, col4 = st.columns(4)

# with col1:
#     st.metric("Total Sentences", len(filtered))

# with col2:
#     if "sentiment" in filtered.columns and not filtered.empty:
#         sentiment_counts = filtered["sentiment"].value_counts(normalize=True) * 100
#         pos = sentiment_counts.get("positive", 0)
#         neu = sentiment_counts.get("neutral", 0)
#         neg = sentiment_counts.get("negative", 0)
#         st.metric("Positive / Neutral / Negative (%)", f"{pos:.1f} / {neu:.1f} / {neg:.1f}")
#     else:
#         st.metric("Positive / Neutral / Negative (%)", "–")

# with col3:
#     if "confidence" in filtered.columns and not filtered.empty:
#         st.metric("Avg. Model Confidence", f"{filtered['confidence'].mean():.2f}")
#     else:
#         st.metric("Avg. Model Confidence", "–")

# with col4:
#     if "materiality" in filtered.columns and not filtered.empty:
#         mat_counts = filtered["materiality"].value_counts()
#         top_mat = mat_counts.index[0]
#         st.metric("Most Common Materiality", top_mat)
#     else:
#         st.metric("Most Common Materiality", "–")


# # -------------------------------------------------------
# # Charts
# # -------------------------------------------------------
# tab1, tab2, tab3 = st.tabs(["📊 Distributions", "📌 Top Aspects", "📄 Sentences & Context"])

# with tab1:
#     sub_col1, sub_col2 = st.columns(2)

#     with sub_col1:
#         st.subheader("Sentiment Distribution")
#         if "sentiment" in filtered.columns and not filtered.empty:
#             sentiment_counts = filtered["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
#             sentiment_counts = sentiment_counts.set_index("sentiment")
#             st.bar_chart(sentiment_counts)
#         else:
#             st.write("No sentiment data available.")

#     with sub_col2:
#         st.subheader("Aspect Category Distribution")
#         if "aspect_category" in filtered.columns and not filtered.empty:
#             ac_counts = filtered["aspect_category"].value_counts().rename_axis("aspect_category").reset_index(name="count")
#             ac_counts = ac_counts.set_index("aspect_category")
#             st.bar_chart(ac_counts)
#         else:
#             st.write("No aspect category data available.")

#     st.subheader("Materiality vs Impact Level")
#     if {"materiality", "impact_level"}.issubset(filtered.columns) and not filtered.empty:
#         cross = pd.crosstab(filtered["materiality"], filtered["impact_level"])
#         st.dataframe(cross)
#     else:
#         st.write("Not enough data to display this table.")


# with tab2:
#     st.subheader("Top Aspects by Frequency")

#     if "aspect" in filtered.columns and not filtered.empty:
#         top_n = st.slider("Number of top aspects", 3, 20, 10)
#         aspect_counts = (
#             filtered["aspect"]
#             .value_counts()
#             .head(top_n)
#             .rename_axis("aspect")
#             .reset_index(name="count")
#         )
#         aspect_counts = aspect_counts.set_index("aspect")
#         st.bar_chart(aspect_counts)
#         st.dataframe(aspect_counts.reset_index())
#     else:
#         st.write("No aspect data available.")


# with tab3:
#     st.subheader("Sentence-level View")

#     # Columns to show in table
#     cols_for_table = [
#         c
#         for c in [
#             "sentence",
#             "aspect",
#             "aspect_category",
#             "sentiment",
#             "sentiment_score",
#             "tone",
#             "materiality",
#             "stakeholder",
#             "esg_risk_type",
#             "impact_level",
#             "time_horizon",
#             "claim_type",
#             "has_kpi",
#             "has_target",
#             "ontology_uri",
#             "regulation_uri",
#             "value_chain_stage",
#             "page_number",
#             "filename",
#         ]
#         if c in filtered.columns
#     ]

#     st.dataframe(filtered[cols_for_table])

#     # Context from the report
#     with st.expander("📖 Show report context (markdown_full) for current file(s)"):
#         if "markdown_full" in filtered.columns:
#             # Show one block per unique document
#             for fname, group in filtered.groupby("filename"):
#                 st.markdown(f"### 📄 {fname}")
#                 # Show first markdown_full for this file
#                 sample_context = group["markdown_full"].dropna().iloc[0]
#                 st.markdown(sample_context)
#         else:
#             st.write("No markdown_full column found.")

#     with st.expander("🧾 Show raw original text column"):
#         if "original" in filtered.columns:
#             for fname, group in filtered.groupby("filename"):
#                 st.markdown(f"### 📄 {fname}")
#                 sample_orig = group["original"].dropna().iloc[0]
#                 st.markdown(sample_orig)
#         else:
#             st.write("No original column found.")
