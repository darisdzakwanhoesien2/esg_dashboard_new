# 5_Tone_Distribution.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Tone Distribution Explorer", layout="wide")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
st.title("📊 Tone Distribution Explorer")

st.write("Visualize tone distribution computed from Aspect × Sentiment pairs.")

# Adjust to your actual path
# AGG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "tone_distribution.csv")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

AGG_PATH = os.path.join(PROJECT_ROOT, "data", "output_in_csv.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(AGG_PATH)
    df.columns = df.columns.str.lower().str.strip()
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load file: {AGG_PATH}")
    st.stop()

st.success(f"Loaded {len(df)} rows")


# ---------------------------------------------------------
# PARSE DATA DISTRIBUTION
# ---------------------------------------------------------
def parse_distribution(x):
    if not isinstance(x, str):
        return []
    return [int(v.strip()) for v in x.split(",") if v.strip().isdigit()]

df["data_distribution_list"] = df["data distribution"].apply(parse_distribution)


# ---------------------------------------------------------
# SIDEBAR MULTI-SELECT FILTERS
# ---------------------------------------------------------
st.sidebar.header("Filters (Multi-select)")

aspect_options = sorted(df["aspect_category"].dropna().unique())
sentiment_options = sorted(df["sentiment"].dropna().unique())
tone_options = sorted(df["minimum_tone"].dropna().unique())

selected_aspects = st.sidebar.multiselect(
    "Aspect Category",
    aspect_options,
    default=aspect_options
)

selected_sentiments = st.sidebar.multiselect(
    "Sentiment",
    sentiment_options,
    default=sentiment_options
)

selected_tones = st.sidebar.multiselect(
    "Minimum Tone",
    tone_options,
    default=tone_options
)

# Filtering logic
df_filtered = df[
    df["aspect_category"].isin(selected_aspects) &
    df["sentiment"].isin(selected_sentiments) &
    df["minimum_tone"].isin(selected_tones)
]

st.write(f"### Filtered Rows: {len(df_filtered)}")
st.dataframe(df_filtered, use_container_width=True)


# ---------------------------------------------------------
# PIE CHART: DATA DISTRIBUTION
# ---------------------------------------------------------
st.markdown("### 🥧 Data Distribution Pie Chart")

if df_filtered.empty:
    st.info("No data after filters")
else:
    # Combine multiple rows by summing
    combined_dist = []
    for lst in df_filtered["data_distribution_list"]:
        if not combined_dist:
            combined_dist = lst.copy()
        else:
            # match lengths
            if len(lst) > len(combined_dist):
                combined_dist.extend([0] * (len(lst) - len(combined_dist)))
            for i in range(len(lst)):
                combined_dist[i] += lst[i]

    if len(combined_dist) > 0:
        labels = [f"Tone {i+1}" for i in range(len(combined_dist))]
        fig = px.pie(values=combined_dist, names=labels, hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Tone distribution is empty.")


# ---------------------------------------------------------
# BAR CHART: MINIMUM AMOUNT
# ---------------------------------------------------------
st.markdown("### 📦 Minimum Amount Bar Chart")

if len(df_filtered):
    fig2 = px.bar(
        df_filtered,
        x="minimum_tone",
        y="minimum_amount",
        color="minimum_tone",
        title="Minimum Tone Counts",
        labels={"minimum_tone": "Tone", "minimum_amount": "Count"}
    )
    st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------
# DESCENDING ORDER CHART
# ---------------------------------------------------------
st.markdown("### 📉 Descending Tone Order")

if "descending order" in df_filtered.columns and len(df_filtered):
    df_filtered["descending_list"] = df_filtered["descending order"].apply(
        lambda x: [v.strip() for v in str(x).split(",") if v.strip()]
    )

    # Only show first row for now
    first_list = df_filtered["descending_list"].iloc[0]

    fig3 = go.Figure(data=[go.Bar(
        x=list(range(1, len(first_list)+1)),
        y=[1]*len(first_list),
        text=first_list,
        textposition="inside"
    )])
    fig3.update_layout(
        title="Descending Tone Ranking",
        xaxis_title="Rank",
        yaxis=dict(showticklabels=False)
    )
    st.plotly_chart(fig3, use_container_width=True)


# ---------------------------------------------------------
# MINI SANKEY (Aspect → Sentiment → Minimum Tone)
# ---------------------------------------------------------
st.markdown("### 🔗 Mini Sankey Diagram")

if len(df_filtered):
    # Use aggregated values
    sankey_data = df_filtered.groupby(["aspect_category", "sentiment", "minimum_tone"])["minimum_amount"].sum().reset_index()

    nodes = list(pd.unique(sankey_data[["aspect_category", "sentiment", "minimum_tone"]].values.ravel()))
    node_index = {n: i for i, n in enumerate(nodes)}

    src = sankey_data["aspect_category"].map(node_index).tolist() + \
          sankey_data["sentiment"].map(node_index).tolist()

    tgt = sankey_data["sentiment"].map(node_index).tolist() + \
          sankey_data["minimum_tone"].map(node_index).tolist()

    value = sankey_data["minimum_amount"].tolist() * 2

    fig4 = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(label=nodes, pad=15, thickness=18),
        link=dict(source=src, target=tgt, value=value)
    )])
    fig4.update_layout(title="Aspect → Sentiment → Minimum Tone Flow")
    st.plotly_chart(fig4, use_container_width=True)


# ---------------------------------------------------------
# HEATMAP VIEW
# ---------------------------------------------------------
st.markdown("### 🔥 Heatmap: Minimum Amount by Aspect × Sentiment")

pivot = df_filtered.pivot_table(
    index="aspect_category",
    columns="sentiment",
    values="minimum_amount",
    aggfunc="sum"
)

if pivot.empty:
    st.info("Heatmap cannot be generated.")
else:
    fig5 = px.imshow(
        pivot,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig5, use_container_width=True)
