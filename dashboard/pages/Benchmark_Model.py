import streamlit as st

from ui.sidebar import render_sidebar
from ui.text_input import render_text_input
from ui.results import render_results

from services.hf_loader import load_pipeline
from services.inference import run_inference
from utils.env import get_hf_token

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="🌍 ESG & Climate NLP Tester",
    layout="wide"
)

st.title("🌍 ESG & Climate NLP Model Tester")
st.caption("Test one text against ClimateBERT & ESGBERT models")

HF_TOKEN = get_hf_token()

if not HF_TOKEN:
    st.warning(
        "⚠️ Hugging Face token not found. "
        "Some models require authentication. "
        "Set HF_TOKEN or run `huggingface-cli login`."
    )

# --------------------------------------------------
# UI
# --------------------------------------------------
selected_models = render_sidebar()
text = render_text_input()

run = st.button("🚀 Run Models")

# --------------------------------------------------
# Inference
# --------------------------------------------------
if run:
    if not text.strip():
        st.warning("Please enter some text.")
    elif not selected_models:
        st.warning("Please select at least one model.")
    else:
        rows = []

        with st.spinner("Running inference..."):
            for m in selected_models:
                clf, err = load_pipeline(m["id"])

                if err:
                    rows.append({
                        "model_name": m["name"],
                        "model_id": m["id"],
                        "label": "LOAD_ERROR",
                        "score": None,
                        "error": err
                    })
                    continue

                outputs, err = run_inference(clf, text)

                if err:
                    rows.append({
                        "model_name": m["name"],
                        "model_id": m["id"],
                        "label": "INFER_ERROR",
                        "score": None,
                        "error": err
                    })
                else:
                    for o in outputs:
                        rows.append({
                            "model_name": m["name"],
                            "model_id": m["id"],
                            "label": o["label"],
                            "score": o["score"],
                            "error": None
                        })

        render_results(rows)
