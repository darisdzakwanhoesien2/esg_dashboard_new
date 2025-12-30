import streamlit as st
import pandas as pd
from utils.data_loader import load_and_parse
from utils.aspect_clustering import cluster_aspect

st.set_page_config(layout="wide")
st.title("🔍 Aspect Mapping — Before vs After")

df = load_and_parse()

if df.empty or "aspect" not in df.columns:
    st.warning("No aspect data available.")
    st.stop()

df = df.copy()
df["aspect_cluster"] = df["aspect"].apply(cluster_aspect)

comparison = (
    df[["aspect", "aspect_cluster"]]
    .value_counts()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

st.dataframe(comparison, use_container_width=True)


# import streamlit as st
# import pandas as pd
# from utils.aspect_clustering import cluster_aspect

# st.set_page_config(layout="wide")
# st.title("🔍 Aspect Mapping — Before vs After")

# df = st.session_state["filtered_df"].copy()

# df["aspect_cluster"] = df["aspect"].apply(cluster_aspect)

# comparison = (
#     df[["aspect", "aspect_cluster"]]
#     .value_counts()
#     .reset_index(name="count")
#     .sort_values("count", ascending=False)
# )

# st.dataframe(
#     comparison,
#     use_container_width=True
# )
