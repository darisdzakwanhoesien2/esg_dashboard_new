# 7_Upload_Balanced_Tool.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(page_title="Upload-Based Tone Balancer", layout="wide")

st.title("📤 Upload-Based Tone Distribution & Balancer")
st.write("Upload a dataset and automatically generate tone distribution, charts, and balanced sampling.")

# ----------------------------------------------------------
# FILE UPLOADER
# ----------------------------------------------------------
st.header("1️⃣ Upload Your Dataset")

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if not uploaded:
    st.info("Please upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded)
df.columns = df.columns.str.lower().str.strip()

required_cols = ["aspect_category", "sentiment", "tone"]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

st.success(f"File uploaded successfully. Loaded {len(df)} rows.")

# ----------------------------------------------------------
# COMPUTE TONE DISTRIBUTION TABLE (same as page 6)
# ----------------------------------------------------------

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

tone_df = compute_tone_distribution(df)

st.header("2️⃣ Generated Tone Distribution Table")
st.dataframe(tone_df, use_container_width=True)

# Parse distribution string → list of ints
def parse_dist(x):
    if not isinstance(x, str):
        return []
    return [int(v.strip()) for v in x.split(",") if v.strip().isdigit()]

tone_df["data_distribution_list"] = tone_df["data_distribution"].apply(parse_dist)

# ----------------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------------
st.sidebar.header("Filters")

aspect_options = sorted(tone_df["aspect_category"].unique())
sentiment_options = sorted(tone_df["sentiment"].unique())
tone_options = sorted(tone_df["minimum_tone"].unique())

selected_aspects = st.sidebar.multiselect("Aspect Category", aspect_options, default=aspect_options)
selected_sentiments = st.sidebar.multiselect("Sentiment", sentiment_options, default=sentiment_options)
selected_tones = st.sidebar.multiselect("Minimum Tone", tone_options, default=tone_options)

filtered = tone_df[
    tone_df["aspect_category"].isin(selected_aspects) &
    tone_df["sentiment"].isin(selected_sentiments) &
    tone_df["minimum_tone"].isin(selected_tones)
]

st.write(f"### Filtered Rows: {len(filtered)}")
st.dataframe(filtered, use_container_width=True)

# ----------------------------------------------------------
# COMBINED TONE DISTRIBUTION
# ----------------------------------------------------------
st.markdown("## 🥧 Combined Tone Distribution")

if filtered.empty:
    st.info("No groups match your filters.")
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

# ----------------------------------------------------------
# BAR CHART
# ----------------------------------------------------------
st.markdown("## 📦 Minimum Amount by Tone")

if not filtered.empty:
    barfig = px.bar(filtered, x="minimum_tone", y="minimum_amount", color="minimum_tone")
    st.plotly_chart(barfig, use_container_width=True)

# ----------------------------------------------------------
# DESCENDING ORDER
# ----------------------------------------------------------
st.markdown("## 📉 Tone Ranking Example")

if not filtered.empty:
    first = filtered.iloc[0]
    ranks = [x.strip() for x in first["descending_order"].split(",")]

    fig_rank = go.Figure(
        data=[go.Bar(x=list(range(1, len(ranks) + 1)), y=[1]*len(ranks), text=ranks, textposition="inside")]
    )
    fig_rank.update_layout(xaxis_title="Rank", yaxis=dict(showticklabels=False))
    st.plotly_chart(fig_rank, use_container_width=True)

# ----------------------------------------------------------
# MINI SANKEY
# ----------------------------------------------------------
st.markdown("## 🔗 Aspect → Sentiment → Tone Sankey")

if not filtered.empty:
    sankey_data = filtered.groupby(["aspect_category", "sentiment", "minimum_tone"])["minimum_amount"].sum().reset_index()

    nodes = list(pd.unique(sankey_data[["aspect_category", "sentiment", "minimum_tone"]].values.ravel()))
    idx = {n: i for i, n in enumerate(nodes)}

    src = sankey_data["aspect_category"].map(idx).tolist() + sankey_data["sentiment"].map(idx).tolist()
    tgt = sankey_data["sentiment"].map(idx).tolist() + sankey_data["minimum_tone"].map(idx).tolist()
    val = sankey_data["minimum_amount"].tolist() * 2

    fig_sankey = go.Figure(
        data=[go.Sankey(
            node=dict(label=nodes, pad=15, thickness=18),
            link=dict(source=src, target=tgt, value=val)
        )]
    )
    st.plotly_chart(fig_sankey, use_container_width=True)

# ----------------------------------------------------------
# BALANCED DATASET EXPORT
# ----------------------------------------------------------
st.markdown("## 🎯 Balanced Dataset Export")

sample_n = st.number_input(
    "Number of samples per group",
    min_value=1,
    max_value=2000,
    value=50,
    step=1
)

if filtered.empty:
    st.info("No groups to export.")
else:
    balanced_rows = []

    for _, row in filtered.iterrows():
        aspect = row["aspect_category"]
        sentiment = row["sentiment"]
        tone = row["minimum_tone"]

        subset = df[
            (df["aspect_category"] == aspect) &
            (df["sentiment"] == sentiment) &
            (df["tone"] == tone)
        ]

        if subset.empty:
            continue

        if len(subset) >= sample_n:
            sampled = subset.sample(sample_n, replace=False, random_state=42)
        else:
            sampled = subset.sample(sample_n, replace=True, random_state=42)

        sampled["target_aspect"] = aspect
        sampled["target_sentiment"] = sentiment
        sampled["target_tone"] = tone

        balanced_rows.append(sampled)

    if balanced_rows:
        balanced_df = pd.concat(balanced_rows, ignore_index=True)
        st.success(f"Balanced dataset created with {len(balanced_df)} rows.")

        st.dataframe(balanced_df.head(20), use_container_width=True)

        csv_bytes = balanced_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download Balanced Dataset (CSV)",
            data=csv_bytes,
            file_name=f"balanced_dataset_{sample_n}_per_group.csv",
            mime="text/csv"
        )
    else:
        st.warning("No matching rows found for sampling.")
