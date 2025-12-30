# ==========================================================
# 📊 Tone Distribution Explorer (Ontology-Aware, Path-Fixed)
# ==========================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(page_title="Tone Distribution Explorer", layout="wide")

st.title("📊 Tone Distribution Explorer")
st.write(
    "Ontology-normalized tone distribution computed directly from "
    "`output_in_csv.csv`."
)

# ----------------------------------------------------------
# PATH RESOLUTION (FIXED)
# ----------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve()

DASHBOARD_DIR = CURRENT_DIR.parents[1]          # dashboard/
PROJECT_ROOT = CURRENT_DIR.parents[2]           # project root

ONTOLOGY_DIR = DASHBOARD_DIR / "data"           # dashboard/data/
CSV_DIR = PROJECT_ROOT / "data"                 # root/data/

MASTER_PATH = CSV_DIR / "output_in_csv.csv"

# ----------------------------------------------------------
# LOAD ONTOLOGIES (CORRECT LOCATION)
# ----------------------------------------------------------
try:
    with open(ONTOLOGY_DIR / "aspect_category_ontology.json") as f:
        ASPECT_ONTOLOGY = json.load(f)

    with open(ONTOLOGY_DIR / "sentiment_ontology.json") as f:
        SENTIMENT_ONTOLOGY = json.load(f)

    with open(ONTOLOGY_DIR / "tone_ontology.json") as f:
        TONE_ONTOLOGY = json.load(f)

except FileNotFoundError as e:
    st.error(f"❌ Ontology file not found:\n{e}")
    st.stop()


def build_alias_map(ontology):
    mapping = {}
    for canonical, meta in ontology.items():
        for alias in meta.get("aliases", []):
            if alias is not None:
                mapping[str(alias).strip().lower()] = canonical
    return mapping


ASPECT_MAP = build_alias_map(ASPECT_ONTOLOGY)
SENTIMENT_MAP = build_alias_map(SENTIMENT_ONTOLOGY)
TONE_MAP = build_alias_map(TONE_ONTOLOGY)


def normalize(value, mapping):
    if pd.isna(value):
        return "OTHER"
    return mapping.get(str(value).strip().lower(), "OTHER")


# ----------------------------------------------------------
# LOAD MASTER DATASET
# ----------------------------------------------------------
@st.cache_data
def load_master():
    df = pd.read_csv(MASTER_PATH)
    df.columns = df.columns.str.lower().str.strip()
    return df


try:
    raw = load_master()
except Exception as e:
    st.error(f"❌ Failed to load dataset {MASTER_PATH}:\n{e}")
    st.stop()

st.success(f"✅ Loaded {len(raw)} rows from output_in_csv.csv")

# ----------------------------------------------------------
# VALIDATE REQUIRED COLUMNS
# ----------------------------------------------------------
required_cols = {"aspect_category", "sentiment", "tone"}
missing = required_cols - set(raw.columns)
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# ----------------------------------------------------------
# NORMALIZE USING ONTOLOGIES
# ----------------------------------------------------------
raw["aspect_norm"] = raw["aspect_category"].apply(lambda x: normalize(x, ASPECT_MAP))
raw["sentiment_norm"] = raw["sentiment"].apply(lambda x: normalize(x, SENTIMENT_MAP))
raw["tone_norm"] = raw["tone"].apply(lambda x: normalize(x, TONE_MAP))

# ----------------------------------------------------------
# COMPUTE TONE DISTRIBUTION (CORRECT MINIMUM LOGIC)
# ----------------------------------------------------------
@st.cache_data
def compute_tone_distribution(df):
    rows = []

    for (aspect, sentiment), g in df.groupby(["aspect_norm", "sentiment_norm"]):
        counts = g["tone_norm"].value_counts()

        if counts.empty:
            continue

        rows.append({
            "aspect_category": aspect,
            "sentiment": sentiment,
            "minimum_tone": counts.idxmin(),
            "minimum_amount": int(counts.min()),
            "group_size": len(g)
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["aspect_category", "sentiment"])
        .reset_index(drop=True)
    )


tone_df = compute_tone_distribution(raw)

st.subheader("Auto-Generated Tone Distribution")
st.dataframe(tone_df, use_container_width=True)

# ----------------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------------
st.sidebar.header("Filters")

f_aspects = st.sidebar.multiselect(
    "Aspect Category",
    sorted(tone_df["aspect_category"].unique()),
    default=sorted(tone_df["aspect_category"].unique())
)

f_sentiments = st.sidebar.multiselect(
    "Sentiment",
    sorted(tone_df["sentiment"].unique()),
    default=sorted(tone_df["sentiment"].unique())
)

f_tones = st.sidebar.multiselect(
    "Minimum Tone",
    sorted(tone_df["minimum_tone"].unique()),
    default=sorted(tone_df["minimum_tone"].unique())
)

filtered = tone_df[
    tone_df["aspect_category"].isin(f_aspects) &
    tone_df["sentiment"].isin(f_sentiments) &
    tone_df["minimum_tone"].isin(f_tones)
]

st.write(f"### Filtered Rows: {len(filtered)}")
st.dataframe(filtered, use_container_width=True)

# ----------------------------------------------------------
# PIE — COMBINED MINIMUM TONE
# ----------------------------------------------------------
st.markdown("## 🥧 Combined Minimum Tone Distribution")

if not filtered.empty:
    pie_df = (
        filtered.groupby("minimum_tone")["minimum_amount"]
        .sum()
        .reset_index()
    )

    fig_pie = px.pie(
        pie_df,
        names="minimum_tone",
        values="minimum_amount",
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ----------------------------------------------------------
# BAR — MINIMUM AMOUNT
# ----------------------------------------------------------
st.markdown("## 📦 Minimum Amount per Tone")

fig_bar = px.bar(
    filtered,
    x="minimum_tone",
    y="minimum_amount",
    color="minimum_tone"
)
st.plotly_chart(fig_bar, use_container_width=True)

# ----------------------------------------------------------
# SANKEY — NAMESPACED (NO COLLISIONS)
# ----------------------------------------------------------
st.markdown("## 🔗 Sankey: Aspect → Sentiment → Minimum Tone")

if not filtered.empty:
    sankey_df = (
        filtered.groupby(["aspect_category", "sentiment", "minimum_tone"])["minimum_amount"]
        .sum()
        .reset_index()
    )

    sankey_df["A"] = sankey_df["aspect_category"].apply(lambda x: f"A:{x}")
    sankey_df["S"] = sankey_df["sentiment"].apply(lambda x: f"S:{x}")
    sankey_df["T"] = sankey_df["minimum_tone"].apply(lambda x: f"T:{x}")

    nodes = pd.unique(sankey_df[["A", "S", "T"]].values.ravel()).tolist()
    idx = {n: i for i, n in enumerate(nodes)}

    source = sankey_df["A"].map(idx).tolist() + sankey_df["S"].map(idx).tolist()
    target = sankey_df["S"].map(idx).tolist() + sankey_df["T"].map(idx).tolist()
    value = sankey_df["minimum_amount"].tolist() * 2

    labels = [
        n.replace("A:", "Aspect: ")
         .replace("S:", "Sentiment: ")
         .replace("T:", "Tone: ")
        for n in nodes
    ]

    fig_sankey = go.Figure(
        data=[go.Sankey(
            node=dict(label=labels, pad=15, thickness=18),
            link=dict(source=source, target=target, value=value)
        )]
    )

    st.plotly_chart(fig_sankey, use_container_width=True)

# ----------------------------------------------------------
# HEATMAP
# ----------------------------------------------------------
st.markdown("## 🔥 Heatmap: Aspect × Sentiment")

pivot = filtered.pivot_table(
    index="aspect_category",
    columns="sentiment",
    values="minimum_amount",
    aggfunc="sum",
    fill_value=0
)

fig_heat = px.imshow(pivot, text_auto=True, color_continuous_scale="Blues")
st.plotly_chart(fig_heat, use_container_width=True)

# ----------------------------------------------------------
# EXPORT
# ----------------------------------------------------------
st.markdown("## ⤵ Download Tone Distribution Table")

st.download_button(
    "Download tone_distribution.csv",
    tone_df.to_csv(index=False).encode("utf-8"),
    file_name="tone_distribution.csv",
    mime="text/csv"
)

st.caption("Computed at runtime using dashboard/data ontologies.")


# # ==========================================================
# # 📊 Tone Distribution Explorer (Ontology-Aware, Fixed)
# # ==========================================================

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import json
# from pathlib import Path

# # ----------------------------------------------------------
# # PAGE CONFIG
# # ----------------------------------------------------------
# st.set_page_config(page_title="Tone Distribution Explorer", layout="wide")

# st.title("📊 Tone Distribution Explorer")
# st.write(
#     "Ontology-normalized tone distribution computed directly from "
#     "`output_in_csv.csv`."
# )

# # ----------------------------------------------------------
# # PATHS
# # ----------------------------------------------------------
# CURRENT_DIR = Path(__file__).resolve()
# PROJECT_ROOT = CURRENT_DIR.parents[2]
# DATA_DIR = PROJECT_ROOT / "data"

# MASTER_PATH = DATA_DIR / "output_in_csv.csv"

# # ----------------------------------------------------------
# # LOAD ONTOLOGIES
# # ----------------------------------------------------------
# with open(DATA_DIR / "aspect_category_ontology.json") as f:
#     ASPECT_ONTOLOGY = json.load(f)

# with open(DATA_DIR / "sentiment_ontology.json") as f:
#     SENTIMENT_ONTOLOGY = json.load(f)

# with open(DATA_DIR / "tone_ontology.json") as f:
#     TONE_ONTOLOGY = json.load(f)


# def build_alias_map(ontology):
#     m = {}
#     for canonical, meta in ontology.items():
#         for a in meta.get("aliases", []):
#             if a is not None:
#                 m[str(a).strip().lower()] = canonical
#     return m


# ASPECT_MAP = build_alias_map(ASPECT_ONTOLOGY)
# SENTIMENT_MAP = build_alias_map(SENTIMENT_ONTOLOGY)
# TONE_MAP = build_alias_map(TONE_ONTOLOGY)


# def normalize(val, mapping):
#     if pd.isna(val):
#         return "OTHER"
#     return mapping.get(str(val).strip().lower(), "OTHER")


# # ----------------------------------------------------------
# # LOAD MASTER DATA
# # ----------------------------------------------------------
# @st.cache_data
# def load_master():
#     df = pd.read_csv(MASTER_PATH)
#     df.columns = df.columns.str.lower().str.strip()
#     return df


# try:
#     raw = load_master()
# except Exception as e:
#     st.error(f"Failed to load dataset: {e}")
#     st.stop()

# st.success(f"Loaded {len(raw)} rows from output_in_csv.csv")

# # ----------------------------------------------------------
# # VALIDATE COLUMNS
# # ----------------------------------------------------------
# required_cols = {"aspect_category", "sentiment", "tone"}
# missing = required_cols - set(raw.columns)
# if missing:
#     st.error(f"Missing required columns: {missing}")
#     st.stop()

# # ----------------------------------------------------------
# # NORMALIZE USING ONTOLOGIES
# # ----------------------------------------------------------
# raw["aspect_norm"] = raw["aspect_category"].apply(lambda x: normalize(x, ASPECT_MAP))
# raw["sentiment_norm"] = raw["sentiment"].apply(lambda x: normalize(x, SENTIMENT_MAP))
# raw["tone_norm"] = raw["tone"].apply(lambda x: normalize(x, TONE_MAP))

# # ----------------------------------------------------------
# # COMPUTE TONE DISTRIBUTION (CORRECT MINIMUM)
# # ----------------------------------------------------------
# @st.cache_data
# def compute_tone_distribution(df):
#     rows = []

#     for (a, s), g in df.groupby(["aspect_norm", "sentiment_norm"]):
#         counts = g["tone_norm"].value_counts()

#         if counts.empty:
#             continue

#         min_tone = counts.idxmin()
#         min_amount = int(counts.min())

#         rows.append({
#             "aspect_category": a,
#             "sentiment": s,
#             "minimum_tone": min_tone,
#             "minimum_amount": min_amount,
#             "group_size": len(g)
#         })

#     return (
#         pd.DataFrame(rows)
#         .sort_values(["aspect_category", "sentiment"])
#         .reset_index(drop=True)
#     )


# tone_df = compute_tone_distribution(raw)

# st.subheader("Auto-Generated Tone Distribution")
# st.dataframe(tone_df, use_container_width=True)

# # ----------------------------------------------------------
# # SIDEBAR FILTERS
# # ----------------------------------------------------------
# st.sidebar.header("Filters")

# f_aspects = st.sidebar.multiselect(
#     "Aspect Category",
#     sorted(tone_df["aspect_category"].unique()),
#     default=sorted(tone_df["aspect_category"].unique())
# )

# f_sentiments = st.sidebar.multiselect(
#     "Sentiment",
#     sorted(tone_df["sentiment"].unique()),
#     default=sorted(tone_df["sentiment"].unique())
# )

# f_tones = st.sidebar.multiselect(
#     "Minimum Tone",
#     sorted(tone_df["minimum_tone"].unique()),
#     default=sorted(tone_df["minimum_tone"].unique())
# )

# filtered = tone_df[
#     tone_df["aspect_category"].isin(f_aspects) &
#     tone_df["sentiment"].isin(f_sentiments) &
#     tone_df["minimum_tone"].isin(f_tones)
# ]

# st.write(f"### Filtered Rows: {len(filtered)}")
# st.dataframe(filtered, use_container_width=True)

# # ----------------------------------------------------------
# # PIE — COMBINED MINIMUM TONE
# # ----------------------------------------------------------
# st.markdown("## 🥧 Combined Minimum Tone Distribution")

# if not filtered.empty:
#     pie_df = (
#         filtered.groupby("minimum_tone")["minimum_amount"]
#         .sum()
#         .reset_index()
#     )

#     fig_pie = px.pie(
#         pie_df,
#         names="minimum_tone",
#         values="minimum_amount",
#         hole=0.4
#     )
#     st.plotly_chart(fig_pie, use_container_width=True)

# # ----------------------------------------------------------
# # BAR — MINIMUM AMOUNT
# # ----------------------------------------------------------
# st.markdown("## 📦 Minimum Amount per Tone")

# fig_bar = px.bar(
#     filtered,
#     x="minimum_tone",
#     y="minimum_amount",
#     color="minimum_tone"
# )
# st.plotly_chart(fig_bar, use_container_width=True)

# # ----------------------------------------------------------
# # SANKEY — NAMESPACED (FIXED)
# # ----------------------------------------------------------
# st.markdown("## 🔗 Sankey: Aspect → Sentiment → Minimum Tone")

# if not filtered.empty:
#     sankey_df = (
#         filtered.groupby(["aspect_category", "sentiment", "minimum_tone"])["minimum_amount"]
#         .sum()
#         .reset_index()
#     )

#     sankey_df["A"] = sankey_df["aspect_category"].apply(lambda x: f"A:{x}")
#     sankey_df["S"] = sankey_df["sentiment"].apply(lambda x: f"S:{x}")
#     sankey_df["T"] = sankey_df["minimum_tone"].apply(lambda x: f"T:{x}")

#     nodes = pd.unique(sankey_df[["A", "S", "T"]].values.ravel()).tolist()
#     idx = {n: i for i, n in enumerate(nodes)}

#     src = sankey_df["A"].map(idx).tolist() + sankey_df["S"].map(idx).tolist()
#     tgt = sankey_df["S"].map(idx).tolist() + sankey_df["T"].map(idx).tolist()
#     val = sankey_df["minimum_amount"].tolist() * 2

#     labels = [
#         n.replace("A:", "Aspect: ")
#          .replace("S:", "Sentiment: ")
#          .replace("T:", "Tone: ")
#         for n in nodes
#     ]

#     fig_sankey = go.Figure(
#         data=[go.Sankey(
#             node=dict(label=labels, pad=15, thickness=18),
#             link=dict(source=src, target=tgt, value=val)
#         )]
#     )

#     st.plotly_chart(fig_sankey, use_container_width=True)

# # ----------------------------------------------------------
# # HEATMAP
# # ----------------------------------------------------------
# st.markdown("## 🔥 Heatmap: Aspect × Sentiment")

# pivot = filtered.pivot_table(
#     index="aspect_category",
#     columns="sentiment",
#     values="minimum_amount",
#     aggfunc="sum",
#     fill_value=0
# )

# fig_heat = px.imshow(pivot, text_auto=True, color_continuous_scale="Blues")
# st.plotly_chart(fig_heat, use_container_width=True)

# # ----------------------------------------------------------
# # EXPORT TABLE
# # ----------------------------------------------------------
# st.markdown("## ⤵ Download Tone Distribution Table")

# st.download_button(
#     "Download tone_distribution.csv",
#     tone_df.to_csv(index=False).encode("utf-8"),
#     file_name="tone_distribution.csv",
#     mime="text/csv"
# )

# st.caption("Computed at runtime from output_in_csv.csv using ontologies.")
