import streamlit as st
import pandas as pd

def render_results(rows):
    df = pd.DataFrame(rows)

    st.subheader("📊 Model Outputs")
    st.dataframe(df, use_container_width=True)

    if "score" in df.columns:
        st.subheader("🏆 Top Prediction per Model")
        top = (
            df.dropna(subset=["score"])
              .sort_values(["model_id", "score"], ascending=[True, False])
              .groupby("model_id")
              .head(1)
        )
        st.dataframe(top, use_container_width=True)
