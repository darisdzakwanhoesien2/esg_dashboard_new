import streamlit as st
import pandas as pd

from utils.metrics import compute_metrics
from utils.validation import validate_columns

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Prediction Metric Analysis",
    layout="wide"
)

st.title("📊 Prediction Metric Analysis Dashboard")
st.caption("Evaluate Aspect Category, Sentiment, and Tone predictions")

# --------------------------------------------------
# REQUIRED COLUMNS
# --------------------------------------------------
REQUIRED_COLS = [
    "sentence",
    "aspect_category",
    "sentiment",
    "tone"
]

TARGETS = ["aspect_category", "sentiment", "tone"]

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
st.header("1️⃣ Upload Datasets")

col1, col2 = st.columns(2)

with col1:
    gt_file = st.file_uploader("📥 Ground Truth CSV", type=["csv"])

with col2:
    pred_file = st.file_uploader("📤 Prediction CSV", type=["csv"])

if not gt_file or not pred_file:
    st.info("Please upload both Ground Truth and Prediction CSV files.")
    st.stop()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
gt_df = pd.read_csv(gt_file)
pred_df = pd.read_csv(pred_file)

gt_df.columns = gt_df.columns.str.lower().str.strip()
pred_df.columns = pred_df.columns.str.lower().str.strip()

# --------------------------------------------------
# VALIDATION
# --------------------------------------------------
missing_gt = validate_columns(gt_df, REQUIRED_COLS)
missing_pred = validate_columns(pred_df, REQUIRED_COLS)

if missing_gt or missing_pred:
    st.error(
        f"""
        ❌ Missing required columns

        Ground Truth missing: {missing_gt}
        Prediction missing: {missing_pred}
        """
    )
    st.stop()

if len(gt_df) != len(pred_df):
    st.error("❌ Ground Truth and Prediction files must have the same number of rows.")
    st.stop()

st.success(f"Loaded {len(gt_df)} rows successfully.")

# --------------------------------------------------
# METRIC COMPUTATION
# --------------------------------------------------
st.header("2️⃣ Evaluation Metrics")

metric_results = {}

for target in TARGETS:
    metrics = compute_metrics(
        gt_df[target],
        pred_df[target]
    )
    metric_results[target] = metrics

# --------------------------------------------------
# DISPLAY METRICS
# --------------------------------------------------
tabs = st.tabs(
    ["Aspect Category", "Sentiment", "Tone"]
)

for tab, target in zip(tabs, TARGETS):
    with tab:
        st.subheader(f"🎯 {target.replace('_', ' ').title()} Metrics")

        df_metrics = pd.DataFrame(metric_results[target], index=[0]).T
        df_metrics.columns = ["Score"]

        st.dataframe(
            df_metrics.style.format("{:.4f}"),
            use_container_width=True
        )

        st.metric("Accuracy", f"{metric_results[target]['accuracy']:.4f}")
        st.metric("F1 Score", f"{metric_results[target]['f1']:.4f}")

# --------------------------------------------------
# ERROR ANALYSIS
# --------------------------------------------------
st.header("3️⃣ Error Analysis")

selected_target = st.selectbox(
    "Select target for error inspection",
    TARGETS
)

errors = gt_df[gt_df[selected_target] != pred_df[selected_target]].copy()
errors["predicted"] = pred_df.loc[errors.index, selected_target]
errors["actual"] = gt_df.loc[errors.index, selected_target]

st.write(f"❌ Total errors: {len(errors)}")

st.dataframe(
    errors[
        ["sentence", "actual", "predicted"]
    ],
    use_container_width=True
)

# --------------------------------------------------
# CONFIDENCE ANALYSIS (OPTIONAL)
# --------------------------------------------------
if "confidence" in pred_df.columns:
    st.header("4️⃣ Confidence vs Errors")

    errors["confidence"] = pred_df.loc[errors.index, "confidence"]

    st.scatter_chart(
        errors[["confidence"]],
        height=300
    )
