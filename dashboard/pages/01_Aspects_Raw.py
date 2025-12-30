# import streamlit as st
# import pandas as pd

# st.set_page_config(layout="wide")
# st.title("📌 Aspects — Raw Model Output")

# df = st.session_state["filtered_df"]

import streamlit as st
import pandas as pd
from utils.data_loader import load_and_parse

st.set_page_config(layout="wide")
st.title("📌 Aspects — Raw (Before Manual Annotation)")

df = load_and_parse()

if df.empty or "aspect" not in df.columns:
    st.warning("No aspect data available.")
    st.stop()

n = st.slider("Show Top N Raw Aspects", 3, 50, 15)

top_raw = (
    df["aspect"]
    .value_counts()
    .sort_values(ascending=False)
    .head(n)
)

raw_df = top_raw.reset_index()
raw_df.columns = ["aspect", "count"]

raw_df["aspect"] = pd.Categorical(
    raw_df["aspect"],
    categories=raw_df["aspect"].tolist(),
    ordered=True
)

st.bar_chart(raw_df.set_index("aspect")["count"])
st.dataframe(raw_df, use_container_width=True)
