import streamlit as st

def render_text_input():
    st.subheader("✍️ Input Text")

    return st.text_area(
        "Enter ESG / Climate-related text",
        height=200,
        placeholder="Example: We commit to achieving net-zero emissions by 2040..."
    )
