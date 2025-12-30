import streamlit as st
import pandas as pd
from utils.data_loader import load_and_parse
from utils.aspect_clustering import cluster_aspect

st.set_page_config(layout="wide")
st.title("🧩 Aspects — After Manual Clustering")

df = load_and_parse()

if df.empty or "aspect" not in df.columns:
    st.warning("No aspect data available.")
    st.stop()

df = df.copy()
df["aspect_cluster"] = df["aspect"].apply(cluster_aspect)

n = st.slider("Show Top N Aspect Clusters", 3, 30, 10)

top_clusters = (
    df["aspect_cluster"]
    .value_counts()
    .sort_values(ascending=False)
    .head(n)
)

cluster_df = top_clusters.reset_index()
cluster_df.columns = ["aspect_cluster", "count"]

cluster_df["aspect_cluster"] = pd.Categorical(
    cluster_df["aspect_cluster"],
    categories=cluster_df["aspect_cluster"].tolist(),
    ordered=True
)

st.bar_chart(cluster_df.set_index("aspect_cluster")["count"])
st.dataframe(cluster_df, use_container_width=True)

UNCLUSTERED_LABELS = {"Unclustered", "Undefined"}
# -------------------------------------------------------
# 🔍 Unclustered / Undefined Aspect Inspection
# -------------------------------------------------------
unclustered_df = (
    df[df["aspect_cluster"].isin(UNCLUSTERED_LABELS)]
    .groupby("aspect")
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

st.subheader("🚨 Unclustered Aspects (Needs Taxonomy Review)")

if unclustered_df.empty:
    st.success("🎉 No unclustered aspects found.")
else:
    joined_text = "\n".join(
        f"{row.aspect} ({row['count']})"
        for _, row in unclustered_df.iterrows()
    )

    st.text_area(
        label="Unclustered Aspects (one per line)",
        value=joined_text,
        height=400
    )



# import streamlit as st
# import pandas as pd
# from utils.aspect_clustering import cluster_aspect

# st.set_page_config(layout="wide")
# st.title("🧩 Aspects — Manually Clustered")

# df = st.session_state["filtered_df"].copy()

# if "aspect" not in df.columns or df.empty:
#     st.warning("No aspect data available.")
#     st.stop()

# df["aspect_cluster"] = df["aspect"].apply(cluster_aspect)

# n = st.slider(
#     "Show Top N Aspect Clusters",
#     3, 30, 10,
#     key="clustered_aspects_n"
# )

# top_clusters = (
#     df["aspect_cluster"]
#     .value_counts()
#     .sort_values(ascending=False)
#     .head(n)
# )

# cluster_df = top_clusters.reset_index()
# cluster_df.columns = ["aspect_cluster", "count"]

# cluster_df["aspect_cluster"] = pd.Categorical(
#     cluster_df["aspect_cluster"],
#     categories=cluster_df["aspect_cluster"].tolist(),
#     ordered=True
# )

# st.bar_chart(cluster_df.set_index("aspect_cluster")["count"])
# st.dataframe(cluster_df, use_container_width=True)
