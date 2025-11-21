# 5_Tone_Distribution.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Tone Distribution Explorer", layout="wide")

# ---------------------------
# Load master dataset (output_in_csv.csv)
# ---------------------------
st.title("📊 Tone Distribution Explorer (Auto-Generated from output_in_csv.csv)")
st.write("This page automatically computes tone distribution from the raw dataset — no intermediate CSV needed.")

# Resolve correct paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

# Correct location of dataset
MASTER_PATH = os.path.join(PROJECT_ROOT, "data", "output_in_csv.csv")

@st.cache_data
def load_master():
    df = pd.read_csv(MASTER_PATH)
    df.columns = df.columns.str.lower().str.strip()
    return df

try:
    raw = load_master()
except Exception as e:
    st.error(f"Could not load {MASTER_PATH}: {e}")
    st.stop()

st.success(f"Loaded {len(raw)} rows from output_in_csv.csv")

# ---------------------------
# Validate required columns
# ---------------------------
required_cols = ["aspect_category", "sentiment", "tone"]
for col in required_cols:
    if col not in raw.columns:
        st.error(f"Missing required column: {col}")
        st.stop()

# ---------------------------
# AUTO-COMPUTE TONE DISTRIBUTION TABLE
# ---------------------------
@st.cache_data
def compute_tone_distribution(df):
    rows = []
    grouped = df.groupby(["aspect_category", "sentiment"])

    for (aspect, sentiment), g in grouped:
        tone_counts = g["tone"].value_counts()

        if len(tone_counts) == 0:
            continue

        minimum_tone = tone_counts.index[0]
        minimum_amount = int(tone_counts.iloc[0])
        descending_order = ", ".join(tone_counts.index.tolist())
        data_distribution = ", ".join(str(int(x)) for x in tone_counts.tolist())

        rows.append({
            "aspect_category": aspect,
            "sentiment": sentiment,
            "minimum_tone": minimum_tone,
            "minimum_amount": minimum_amount,
            "descending_order": descending_order,
            "data_distribution": data_distribution,
            "group_count": len(g)
        })

    tone_df = pd.DataFrame(rows)
    tone_df = tone_df.sort_values(["aspect_category", "sentiment"]).reset_index(drop=True)
    return tone_df

tone_df = compute_tone_distribution(raw)

st.write("### Auto-generated Tone Distribution Table")
st.dataframe(tone_df.head(20), use_container_width=True)

# ---------------------------
# Parse data_distribution into lists
# ---------------------------
def parse_dist(x):
    if not isinstance(x, str):
        return []
    return [int(v.strip()) for v in x.split(",") if v.strip().isdigit()]

tone_df["data_distribution_list"] = tone_df["data_distribution"].apply(parse_dist)

# ---------------------------
# Sidebar filters (multi-select)
# ---------------------------
st.sidebar.header("Filters")

aspect_options = sorted(tone_df["aspect_category"].unique())
sentiment_options = sorted(tone_df["sentiment"].unique())
tone_options = sorted(tone_df["minimum_tone"].unique())

selected_aspects = st.sidebar.multiselect("Aspect Category", aspect_options, default=aspect_options)
selected_sentiments = st.sidebar.multiselect("Sentiment", sentiment_options, default=sentiment_options)
selected_min_tones = st.sidebar.multiselect("Minimum Tone", tone_options, default=tone_options)

filtered = tone_df[
    tone_df["aspect_category"].isin(selected_aspects) &
    tone_df["sentiment"].isin(selected_sentiments) &
    tone_df["minimum_tone"].isin(selected_min_tones)
]

st.write(f"### Filtered Rows: {len(filtered)}")
st.dataframe(filtered, use_container_width=True)

# ----------------------------------------------------------
# BALANCED DATASET EXPORT TOOL
# ----------------------------------------------------------
st.markdown("## 🎯 Balanced Dataset Export")

# Choose number of samples per group
sample_n = st.number_input(
    "Number of samples per group",
    min_value=1,
    max_value=2000,
    value=78,
    step=1
)

# Check groups inside filtered tone summary
if filtered.empty:
    st.info("No groups available for balanced sampling.")
else:
    st.write(f"### Groups selected: {len(filtered)}")
    st.dataframe(filtered[["aspect_category", "sentiment", "minimum_tone", "minimum_amount"]])

    # Create balanced dataset
    balanced_rows = []

    for _, row in filtered.iterrows():
        aspect = row["aspect_category"]
        sentiment = row["sentiment"]
        tone = row["minimum_tone"]

        # Extract matching raw rows
        subset = raw[
            (raw["aspect_category"] == aspect) &
            (raw["sentiment"] == sentiment) &
            (raw["tone"] == tone)
        ]

        if subset.empty:
            continue

        # Downsample or upsample
        if len(subset) >= sample_n:
            sampled = subset.sample(sample_n, replace=False, random_state=42)
        else:
            sampled = subset.sample(sample_n, replace=True, random_state=42)

        sampled = sampled.copy()
        sampled["target_aspect"] = aspect
        sampled["target_sentiment"] = sentiment
        sampled["target_tone"] = tone
        balanced_rows.append(sampled)

    if balanced_rows:
        balanced_df = pd.concat(balanced_rows, ignore_index=True)
        st.success(f"Balanced dataset created with {len(balanced_df)} rows.")

        st.write("Preview:")
        st.dataframe(balanced_df.head(20), use_container_width=True)

        # Provide CSV Download
        csv_bytes = balanced_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download Balanced Dataset (CSV)",
            data=csv_bytes,
            file_name=f"balanced_dataset_{sample_n}_per_group.csv",
            mime="text/csv"
        )
    else:
        st.warning("No matching raw rows found for the selected groups.")


# ---------------------------
# PIE: Combined tone distribution
# ---------------------------
st.markdown("## 🥧 Combined Tone Distribution")

if filtered.empty:
    st.info("No data after filters.")
else:
    combined = []
    for lst in filtered["data_distribution_list"]:
        if not combined:
            combined = lst.copy()
        else:
            if len(lst) > len(combined):
                combined.extend([0] * (len(lst) - len(combined)))
            for i in range(len(lst)):
                combined[i] += lst[i]

    if sum(combined) > 0:
        labels = [f"Tone {i+1}" for i in range(len(combined))]
        pie = px.pie(values=combined, names=labels, hole=0.4)
        st.plotly_chart(pie, use_container_width=True)
    else:
        st.warning("Distribution is empty.")

# ---------------------------
# BAR: Minimum Amount
# ---------------------------
st.markdown("## 📦 Minimum Amount Bar Chart")

if not filtered.empty:
    fig = px.bar(filtered, x="minimum_tone", y="minimum_amount", color="minimum_tone")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# DESCENDING ORDER
# ---------------------------
st.markdown("## 📉 Tone Ranking (Descending Order)")

if not filtered.empty:
    first = filtered.iloc[0]
    ranks = [x.strip() for x in first["descending_order"].split(",")]
    fig2 = go.Figure(data=[go.Bar(
        x=list(range(1, len(ranks) + 1)),
        y=[1] * len(ranks),
        text=ranks,
        textposition="inside"
    )])
    fig2.update_layout(xaxis_title="Rank", yaxis=dict(showticklabels=False))
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------
# MINI-SANKEY
# ---------------------------
st.markdown("## 🔗 Sankey: Aspect → Sentiment → Minimum Tone")

if not filtered.empty:
    sankey_data = filtered.groupby(["aspect_category", "sentiment", "minimum_tone"])["minimum_amount"].sum().reset_index()

    nodes = list(pd.unique(sankey_data[["aspect_category", "sentiment", "minimum_tone"]].values.ravel()))
    idx = {n: i for i, n in enumerate(nodes)}

    src = sankey_data["aspect_category"].map(idx).tolist() + sankey_data["sentiment"].map(idx).tolist()
    tgt = sankey_data["sentiment"].map(idx).tolist() + sankey_data["minimum_tone"].map(idx).tolist()
    val = sankey_data["minimum_amount"].tolist() * 2

    fig3 = go.Figure(data=[go.Sankey(
        node=dict(label=nodes, pad=15, thickness=18),
        link=dict(source=src, target=tgt, value=val)
    )])
    st.plotly_chart(fig3, use_container_width=True)

# ---------------------------
# HEATMAP
# ---------------------------
st.markdown("## 🔥 Heatmap: Aspect × Sentiment (Minimum Amount)")

pivot = filtered.pivot_table(
    index="aspect_category",
    columns="sentiment",
    values="minimum_amount",
    aggfunc="sum"
).fillna(0)

if pivot.empty:
    st.info("Not enough data.")
else:
    fig4 = px.imshow(pivot, text_auto=True, color_continuous_scale="Blues")
    st.plotly_chart(fig4, use_container_width=True)

# ---------------------------
# Download generated table
# ---------------------------
st.markdown("## ⤵ Download Generated Tone Summary Table")

csv_bytes = tone_df.to_csv(index=False).encode("utf-8")
st.download_button("Download tone_distribution (auto-generated)", data=csv_bytes, file_name="tone_distribution.csv", mime="text/csv")

st.caption("Computed automatically from output_in_csv.csv at runtime.")
