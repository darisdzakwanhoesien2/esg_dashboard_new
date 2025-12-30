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

# -------------------------------------------------------
# 🔐 HARDEN SCHEMA (IMPORTANT)
# -------------------------------------------------------
for col in ["filename", "model"]:
    if col in raw_df.columns:
        raw_df[col] = raw_df[col].astype(str)

# =======================================================
# ROBUST JSON PARSING
# =======================================================
def extract_json_block(text):
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
    return isinstance(d, dict) and "sentence" in d and "aspect" in d


def parse_esg_json(text):
    raw = extract_json_block(text)
    normalized = normalize_json(raw)
    return [x for x in normalized if is_valid_esg_object(x)]


@st.cache_data
def parse_annotations(df):
    df = df.copy()
    df["parsed"] = df["text"].apply(parse_esg_json)
    exploded = df.explode("parsed", ignore_index=True)
    parsed_df = pd.json_normalize(exploded["parsed"])
    meta_cols = [c for c in df.columns if c != "parsed"]
    meta = exploded[meta_cols].reset_index(drop=True)
    return pd.concat([meta, parsed_df], axis=1)


df = parse_annotations(raw_df)
st.success(f"Parsed **{len(df)}** ESG sentence records")

# -------------------------------------------------------
# 🔁 GLOBAL SAFE SORTING HELPER (FIX)
# -------------------------------------------------------
def sorted_unique_str(series):
    return sorted(
        series
        .dropna()
        .astype(str)
        .unique()
    )

# -------------------------------------------------------
# Provider parsing
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
    vals = sorted_unique_str(df[col])
    return st.sidebar.multiselect(label, vals, default=vals)

aspect_cats = make_multiselect("Aspect Category", "aspect_category")
sentiments = make_multiselect("Sentiment", "sentiment")
tones = make_multiselect("Tone", "tone")
materialities = make_multiselect("Materiality", "materiality")
stakeholders = make_multiselect("Stakeholder", "stakeholder")
value_chain_stage = make_multiselect("Value Chain Stage", "value_chain_stage")
time_horizon = make_multiselect("Time Horizon", "time_horizon")

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

st.caption(f"Showing **{len(filtered)}** sentences after filtering.")

# -------------------------------------------------------
# Tabs
# -------------------------------------------------------
# tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
#     "📊 Distributions",
#     "📌 Aspects",
#     "📄 Sentence Table",
#     "🤖 Model Comparison",
#     "LLM Breakdown",
#     "🧮 Model Coverage",
#     "📦 Raw JSON View",
#     "📊 Grounding Audit"
# ])
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "📊 Distributions",
    "📌 Aspects (Raw)",
    "📄 Sentence Table",
    "🤖 Model Comparison",
    "LLM Breakdown",
    "🧮 Model Coverage",
    "📦 Raw JSON View",
    "📊 Grounding Audit",
    "📌 Aspects (Raw)",
    "🧩 Aspect Clustering",
    "🧩 Top Aspect Clusters"
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
# -------------------------------------------------------
# TAB 2 — Aspects (SORTED DESCENDING)
# -------------------------------------------------------
with tab2:
    st.subheader("Top Aspects")

    if "aspect" in filtered.columns:
        n = st.slider(
    "Show Top N",
    3,
    30,
    10,
    key="top_n_tab2"
)


        topA = (
            filtered["aspect"]
            .value_counts()
            .sort_values(ascending=False)
            .head(n)
        )

        topA_df = topA.reset_index()
        topA_df.columns = ["aspect", "count"]

        # 🔐 FORCE ORDER
        topA_df["aspect"] = pd.Categorical(
            topA_df["aspect"],
            categories=topA_df["aspect"].tolist(),
            ordered=True
        )

        st.bar_chart(
            topA_df.set_index("aspect")["count"]
        )

        st.dataframe(topA_df, use_container_width=True)


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
# TAB 4 — Model Comparison (ORIGINAL vs HIGHLIGHTED MARKDOWN)
# -------------------------------------------------------
with tab4:
    st.subheader("🤖 LLM Model Comparison (Grounded & Auditable)")

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
        st.warning(f"Missing models: {', '.join(comp['missing'])}")

    # ---------------------------------------------------
    # Sentence Index (GLOBAL FOR THIS PAGE)
    # ---------------------------------------------------
    sentences = list(dict.fromkeys(subset["sentence"].dropna().tolist()))
    sentence_index = {s: i + 1 for i, s in enumerate(sentences)}

    # ---------------------------------------------------
    # Highlight Helper
    # ---------------------------------------------------
    def highlight_sentences(text, sentence_index):
        if not isinstance(text, str):
            return ""

        highlighted = text
        for sentence, idx in sentence_index.items():
            if sentence in highlighted:
                highlighted = highlighted.replace(
                    sentence,
                    f"<span style='background-color:#fff59d; padding:2px; "
                    f"border-radius:4px; font-weight:500;'>"
                    f"[{idx}] {sentence}"
                    f"</span>"
                )
        return highlighted

    # ---------------------------------------------------
    # Extract Page-Level Markdown (same for all models)
    # ---------------------------------------------------
    row0 = subset.iloc[0]

    md_full = row0.get("markdown_full", "")
    md_clean = row0.get("cleaned_markdown", "")

    # ---------------------------------------------------
    # MARKDOWN VISUALIZATION
    # ---------------------------------------------------
    st.markdown("## 📄 Source Text vs Highlighted ESG Sentences")

    # ---- markdown_full ----
    st.markdown("### 🧾 markdown_full")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original**")
        st.markdown(md_full if md_full else "_No markdown_full_", unsafe_allow_html=True)

    with col2:
        st.markdown("**Highlighted**")
        st.markdown(
            highlight_sentences(md_full, sentence_index),
            unsafe_allow_html=True
        )

    # ---- cleaned_markdown ----
    st.markdown("### ✂️ cleaned_markdown")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Original**")
        st.markdown(md_clean if md_clean else "_No cleaned_markdown_", unsafe_allow_html=True)

    with col4:
        st.markdown("**Highlighted**")
        st.markdown(
            highlight_sentences(md_clean, sentence_index),
            unsafe_allow_html=True
        )

    # ---------------------------------------------------
    # Sentence Legend
    # ---------------------------------------------------
    st.markdown("## 🏷 Sentence Index (Reference)")

    legend_df = pd.DataFrame({
        "Index": [sentence_index[s] for s in sentences],
        "Sentence": sentences
    })

    st.dataframe(legend_df, use_container_width=True)

    # ---------------------------------------------------
    # Sentence-Level Model Comparison
    # ---------------------------------------------------
    st.markdown("## 🔍 Sentence-Level Model Comparison")

    pivot = subset.pivot_table(
        index="sentence",
        columns="model",
        values="sentiment",
        aggfunc="first"
    )

    pivot = ensure_all_models(df_pdf, pivot)
    st.dataframe(pivot, use_container_width=True)

    # # ---------------------------------------------------
    # # Presence Validation
    # # ---------------------------------------------------
    # st.markdown("## ✅ Sentence Grounding Check")


    # ---------------------------------------------------
    # Presence Validation (FIXED & STREAMLIT-SAFE)
    # ---------------------------------------------------
    st.markdown("## ✅ Sentence Grounding Check")

    # Always coerce markdown to strings
    md_full_safe = str(md_full or "")
    md_clean_safe = str(md_clean or "")

    presence_rows = []

    for s in sentences:
        presence_rows.append({
            "index": sentence_index.get(s),
            "sentence": s,
            "in_markdown_full": s in md_full_safe,
            "in_cleaned_markdown": s in md_clean_safe,
        })

    # Always create DataFrame
    presence_df = pd.DataFrame(presence_rows)

    # 🔐 GUARANTEE REQUIRED COLUMNS EXIST
    for col in ["in_markdown_full", "in_cleaned_markdown"]:
        if col not in presence_df.columns:
            presence_df[col] = False

    # SAFE boolean operation
    presence_df["found_anywhere"] = (
        presence_df["in_markdown_full"] |
        presence_df["in_cleaned_markdown"]
    )

    st.dataframe(presence_df, use_container_width=True)

    missing = presence_df[~presence_df["found_anywhere"]]

    if not missing.empty:
        st.warning(
            f"⚠️ {len(missing)} sentences are NOT grounded in the source markdown."
        )
        st.dataframe(
            missing[["index", "sentence"]],
            use_container_width=True
        )
    else:
        st.success("✅ All ESG sentences are grounded in the source text.")


    # presence_rows = []
    # for s in sentences:
    #     presence_rows.append({
    #         "index": sentence_index[s],
    #         "sentence": s,
    #         "in_markdown_full": s in str(md_full),
    #         "in_cleaned_markdown": s in str(md_clean)
    #     })

    # presence_df = pd.DataFrame(presence_rows)
    # presence_df["found_anywhere"] = (
    #     presence_df["in_markdown_full"] |
    #     presence_df["in_cleaned_markdown"]
    # )

    st.dataframe(presence_df, use_container_width=True)

    missing = presence_df[~presence_df["found_anywhere"]]
    if not missing.empty:
        st.warning(
            f"⚠️ {len(missing)} sentences are NOT grounded in the source markdown."
        )
        st.dataframe(
            missing[["index", "sentence"]],
            use_container_width=True
        )
    else:
        st.success("✅ All ESG sentences are grounded in the source text.")

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

    filenames = sorted_unique_str(raw_df["filename"])
    selected_file = st.selectbox("Filename", filenames)

    pages = sorted_unique_str(
        raw_df[raw_df["filename"] == selected_file]["page_number"]
    )
    selected_page = st.selectbox("Page", pages)

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

# -------------------------------------------------------
# TAB 8 — Grounding Audit (FIXED)
# -------------------------------------------------------
with tab8:
    st.subheader("📊 Cross-Document Grounding Audit")

    filenames = sorted_unique_str(filtered["filename"])
    sel_file = st.selectbox("Select Filename", filenames)

    pages = sorted_unique_str(
        filtered[filtered["filename"] == sel_file]["page_number"]
    )
    sel_page = st.selectbox("Select Page", pages)

    page_subset = filtered[
        (filtered["filename"] == sel_file) &
        (filtered["page_number"] == sel_page)
    ]

    if page_subset.empty:
        st.warning("No data for selected file/page.")
        st.stop()

    st.dataframe(page_subset[[
        "sentence", "aspect", "sentiment", "model"
    ]], use_container_width=True)



# -------------------------------------------------------
# TAB 2 — Aspects (SORTED DESCENDING)
# -------------------------------------------------------
with tab9:
    st.subheader("Top Aspects")

    if "aspect" in filtered.columns:
        n = st.slider("Show Top N", 3, 30, 10)

        topA = (
            filtered["aspect"]
            .value_counts()
            .sort_values(ascending=False)
            .head(n)
        )

        topA_df = topA.reset_index()
        topA_df.columns = ["aspect", "count"]

        # 🔐 FORCE ORDER
        topA_df["aspect"] = pd.Categorical(
            topA_df["aspect"],
            categories=topA_df["aspect"].tolist(),
            ordered=True
        )

        st.bar_chart(
            topA_df.set_index("aspect")["count"]
        )

        st.dataframe(topA_df, use_container_width=True)

# --------------------------------------------------
# Aspect Cluster JSON (Manual, Auditable)
# --------------------------------------------------
# --------------------------------------------------
# Aspect Cluster JSON (Manual, Authoritative)
# --------------------------------------------------
# --------------------------------------------------
# Aspect Cluster JSON (Manual, Authoritative)
# --------------------------------------------------
ASPECT_CLUSTER_JSON = {
    "Governance": [
        "governance",
        "corporate governance",
        "board oversight",
        "management oversight"
    ],
    "Emissions": [
        "emissions",
        "carbon",
        "carbon footprint",
        "ghg",
        "greenhouse gas",
        "scope 1",
        "scope 2",
        "scope 3"
    ],
    "Financial Reporting": [
        "financial reporting",
        "financial performance",
        "financial disclosure",
        "financial"
    ],
    "Climate & Energy": [
        "energy",
        "renewable energy",
        "energy efficiency",
        "energy transition"
    ],
    "Community & Social Impact": [
        "stakeholder engagement",
        "community",
        "social impact"
    ],
    "Tax & Compliance": [
        "tax",
        "taxation",
        "compliance"
    ]
}

def relabel_aspect_from_json(aspect, cluster_json):
    if not isinstance(aspect, str):
        return "Unclustered"

    a = aspect.lower()
    for cluster, keywords in cluster_json.items():
        for kw in keywords:
            if kw in a:
                return cluster
    return "Unclustered"


# --------------------------------------------------
# Apply Aspect Clustering (SAFE, NO SIDE EFFECTS)
# --------------------------------------------------
filtered = filtered.copy()

if "aspect" in filtered.columns:
    filtered["aspect_cluster"] = filtered["aspect"].apply(
        lambda x: relabel_aspect_from_json(x, ASPECT_CLUSTER_JSON)
    )
else:
    filtered["aspect_cluster"] = "Unclustered"
# -------------------------------------------------------
# TAB 9 — Aspect Clustering (SAFE, NO st.stop)
# -------------------------------------------------------
# -------------------------------------------------------
# TAB 10 — Top Aspect Clusters (SORTED DESCENDING)
# -------------------------------------------------------
with tab10:
    st.subheader("🧩 Top Aspect Clusters")

    if filtered.empty or "aspect_cluster" not in filtered.columns:
        st.warning("No aspect cluster data available.")
    else:
        n = st.slider(
            "Show Top N Aspect Clusters",
            min_value=3,
            max_value=30,
            value=10,
            key="top_clusters_tab10"  # 🔐 UNIQUE
        )

        topC = (
            filtered["aspect_cluster"]
            .value_counts()
            .sort_values(ascending=False)
            .head(n)
        )

        topC_df = topC.reset_index()
        topC_df.columns = ["aspect_cluster", "count"]

        # 🔐 enforce bar order
        topC_df["aspect_cluster"] = pd.Categorical(
            topC_df["aspect_cluster"],
            categories=topC_df["aspect_cluster"].tolist(),
            ordered=True
        )

        st.bar_chart(topC_df.set_index("aspect_cluster")["count"])
        st.dataframe(topC_df, use_container_width=True)

# -------------------------------------------------------
# TAB 10 — Top Aspect Clusters (SORTED DESCENDING)
# -------------------------------------------------------
with tab11:
    st.subheader("🧩 Top Aspect Clusters")

    if filtered.empty:
        st.warning("No data available after filtering.")
    else:
        n = st.slider(
            "Show Top N Aspect Clusters",
            min_value=3,
            max_value=30,
            value=10,
            key="top_clusters_tab10"  # 🔐 UNIQUE KEY
        )

        topC = (
            filtered["aspect_cluster"]
            .value_counts()
            .sort_values(ascending=False)
            .head(n)
        )

        topC_df = topC.reset_index()
        topC_df.columns = ["aspect_cluster", "count"]

        topC_df["aspect_cluster"] = pd.Categorical(
            topC_df["aspect_cluster"],
            categories=topC_df["aspect_cluster"].tolist(),
            ordered=True
        )

        st.bar_chart(
            topC_df.set_index("aspect_cluster")["count"]
        )
        st.dataframe(topC_df, use_container_width=True)

    # st.subheader("🧩 Top Aspect Clusters")

    # if filtered.empty:
    #     st.warning("No data available after filtering.")
    #     st.stop()

    # n = st.slider(
    #     "Show Top N Aspect Clusters",
    #     min_value=3,
    #     max_value=30,
    #     value=10,
    #     key="top_cluster_n"
    # )

    # topC = (
    #     filtered["aspect_cluster"]
    #     .value_counts()
    #     .sort_values(ascending=False)
    #     .head(n)
    # )

    # topC_df = topC.reset_index()
    # topC_df.columns = ["aspect_cluster", "count"]

    # topC_df["aspect_cluster"] = pd.Categorical(
    #     topC_df["aspect_cluster"],
    #     categories=topC_df["aspect_cluster"].tolist(),
    #     ordered=True
    # )

    # st.bar_chart(topC_df.set_index("aspect_cluster")["count"])
    # st.dataframe(topC_df, use_container_width=True)



# # -------------------------------------------------------
# # TAB 9 — Grounding Audit (FIXED)
# # -------------------------------------------------------
# # -------------------------------------------------------
# # TAB 9 — Aspect Clustering (FIXED & SAFE)
# # -------------------------------------------------------
# with tab9:
#     st.subheader("🧩 Aspect Clustering & Relabeling")

#     # ---------------------------------------------------
#     # Safety checks
#     # ---------------------------------------------------
#     required_cols = ["aspect", "aspect_cluster"]

#     missing_cols = [c for c in required_cols if c not in filtered.columns]
#     if missing_cols:
#         st.error(f"Missing required columns: {missing_cols}")
#         st.stop()

#     if filtered.empty:
#         st.warning("No data available after filtering.")
#         st.stop()

#     # ---------------------------------------------------
#     # Coverage Metrics
#     # ---------------------------------------------------
#     total = len(filtered)
#     clustered = (filtered["aspect_cluster"] != "Unclustered").sum()
#     unclustered_cnt = total - clustered

#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("Total Sentences", total)
#     c2.metric("Clustered", clustered)
#     c3.metric("Unclustered", unclustered_cnt)
#     c4.metric(
#         "Coverage %",
#         f"{(clustered / total * 100):.1f}%" if total else "0%"
#     )

#     st.divider()

#     # ---------------------------------------------------
#     # Cluster Distribution (SORTED & STABLE)
#     # ---------------------------------------------------
#     st.markdown("### 📊 Aspect Cluster Distribution")

#     cluster_counts = (
#         filtered["aspect_cluster"]
#         .value_counts()
#         .sort_values(ascending=False)
#         .reset_index()
#     )
#     cluster_counts.columns = ["aspect_cluster", "count"]

#     # Force order (prevents Streamlit resorting)
#     cluster_counts["aspect_cluster"] = pd.Categorical(
#         cluster_counts["aspect_cluster"],
#         categories=cluster_counts["aspect_cluster"].tolist(),
#         ordered=True
#     )

#     st.bar_chart(
#         cluster_counts.set_index("aspect_cluster")["count"]
#     )

#     st.dataframe(cluster_counts, use_container_width=True)

#     st.divider()

#     # ---------------------------------------------------
#     # Raw → Cluster Mapping Audit
#     # ---------------------------------------------------
#     st.markdown("### 🔍 Aspect → Cluster Mapping Audit")

#     audit_df = (
#         filtered[["aspect", "aspect_cluster"]]
#         .drop_duplicates()
#         .sort_values(["aspect_cluster", "aspect"])
#         .reset_index(drop=True)
#     )

#     st.dataframe(audit_df, use_container_width=True)

#     st.divider()

#     # ---------------------------------------------------
#     # Unclustered Diagnostics
#     # ---------------------------------------------------
#     st.markdown("### ⚠️ Unclustered Aspects")

#     unclustered_df = audit_df[
#         audit_df["aspect_cluster"] == "Unclustered"
#     ]

#     if unclustered_df.empty:
#         st.success("✅ All aspects successfully clustered.")
#     else:
#         st.warning(
#             f"{len(unclustered_df)} unique aspects are not clustered. "
#             "Add keywords in the sidebar to improve coverage."
#         )

#         freq_unclustered = (
#             filtered[filtered["aspect_cluster"] == "Unclustered"]["aspect"]
#             .value_counts()
#             .reset_index()
#         )
#         freq_unclustered.columns = ["aspect", "count"]

#         st.dataframe(freq_unclustered, use_container_width=True)

# # -------------------------------------------------------
# # TAB 10 — Top Aspect Clusters (SORTED DESCENDING)
# # -------------------------------------------------------
# with tab10:
#     st.subheader("🧩 Top Aspect Clusters")

#     # Safety
#     if "aspect_cluster" not in filtered.columns:
#         st.error("Missing column: aspect_cluster")
#         st.stop()

#     if filtered.empty:
#         st.warning("No data available after filtering.")
#         st.stop()

#     # Slider
#     n = st.slider(
#         "Show Top N Aspect Clusters",
#         min_value=3,
#         max_value=30,
#         value=10,
#         key="cluster_top_n"
#     )

#     # Aggregate
#     topC = (
#         filtered["aspect_cluster"]
#         .value_counts()
#         .sort_values(ascending=False)
#         .head(n)
#     )

#     topC_df = topC.reset_index()
#     topC_df.columns = ["aspect_cluster", "count"]

#     # 🔐 FORCE CATEGORY ORDER (critical for Streamlit)
#     topC_df["aspect_cluster"] = pd.Categorical(
#         topC_df["aspect_cluster"],
#         categories=topC_df["aspect_cluster"].tolist(),
#         ordered=True
#     )

#     # Chart
#     st.bar_chart(
#         topC_df.set_index("aspect_cluster")["count"]
#     )

#     # Table
#     st.dataframe(topC_df, use_container_width=True)
