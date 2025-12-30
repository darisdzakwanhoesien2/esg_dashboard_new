import streamlit as st
from config.model_registry import MODEL_REGISTRY

def render_sidebar():
    st.sidebar.header("⚙️ Model Selection")

    selected = []

    for group, models in MODEL_REGISTRY.items():
        with st.sidebar.expander(group):
            for name, meta in models.items():
                if st.checkbox(name, key=meta["id"]):
                    selected.append({
                        "name": name,
                        "id": meta["id"],
                        "requires_auth": meta["requires_auth"]
                    })

    return selected
