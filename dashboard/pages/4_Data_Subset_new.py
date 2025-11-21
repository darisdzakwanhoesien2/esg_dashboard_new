# # 3_Data_Subset.py
# import streamlit as st
# import pandas as pd
# import plotly.graph_objects as go
# import os
# import sys
# from io import BytesIO

# # ---------------------------
# # CONFIG / PATHS / HELPERS
# # ---------------------------
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
# sys.path.append(PROJECT_ROOT)

# DATA_PATH = os.path.join(PROJECT_ROOT, "data", "output_in_csv.csv")
# # Path to the uploaded image from the conversation (developer-provided)
# UPLOADED_IMAGE_PATH = "/mnt/data/Screenshot 2025-11-21 at 10.57.18.png"

# @st.cache_data
# def load_dataset(path=DATA_PATH):
#     df = pd.read_csv(path)
#     df.columns = df.columns.str.lower().str.strip()
#     return df

# def build_sankey_from_df(df_in, top_n_aspects=20, top_n_sentiments=10, top_n_tones=10):
#     """
#     Build Sankey nodes and links for Aspect -> Sentiment -> Tone
#     We will sort nodes by frequency and keep top-N for each layer to avoid clutter.
#     """
#     if df_in.empty:
#         return None

#     # compute frequencies
#     aspect_freq = df_in.groupby("aspect_category").size().sort_values(ascending=False)
#     sentiment_freq = df_in.groupby("sentiment").size().sort_values(ascending=False)
#     tone_freq = df_in.groupby("tone").size().sort_values(ascending=False)

#     top_aspects = aspect_freq.head(top_n_aspects).index.tolist()
#     top_sentiments = sentiment_freq.head(top_n_sentiments).index.tolist()
#     top_tones = tone_freq.head(top_n_tones).index.tolist()

#     # filter to top nodes only (still show links even if some nodes filtered)
#     df_filtered = df_in[
#         df_in["aspect_category"].isin(top_aspects) &
#         df_in["sentiment"].isin(top_sentiments) &
#         df_in["tone"].isin(top_tones)
#     ]

#     if df_filtered.empty:
#         # fallback: use original df but only top nodes lists
#         df_filtered = df_in[
#             df_in["aspect_category"].isin(top_aspects) |
#             df_in["sentiment"].isin(top_sentiments) |
#             df_in["tone"].isin(top_tones)
#         ]

#     # Create ordered node list
#     nodes = top_aspects + [s for s in top_sentiments if s not in top_aspects] + [t for t in top_tones if t not in (top_aspects + top_sentiments)]
#     node_index = {name: i for i, name in enumerate(nodes)}

#     # Build links: Aspect -> Sentiment
#     a_to_s = df_filtered.groupby(["aspect_category", "sentiment"]).size().reset_index(name="value")
#     s_to_t = df_filtered.groupby(["sentiment", "tone"]).size().reset_index(name="value")

#     source = []
#     target = []
#     value = []

#     for _, row in a_to_s.iterrows():
#         a, s, v = row["aspect_category"], row["sentiment"], int(row["value"])
#         if a in node_index and s in node_index:
#             source.append(node_index[a])
#             target.append(node_index[s])
#             value.append(v)

#     for _, row in s_to_t.iterrows():
#         s, t, v = row["sentiment"], row["tone"], int(row["value"])
#         if s in node_index and t in node_index:
#             source.append(node_index[s])
#             target.append(node_index[t])
#             value.append(v)

#     if not nodes or not source:
#         return None

#     fig = go.Figure(data=[go.Sankey(
#         node=dict(
#             label=nodes,
#             pad=15,
#             thickness=18,
#             color=["rgba(50,130,200,0.6)" if i < len(top_aspects) else ("rgba(120,200,100,0.6)" if i < len(top_aspects)+len(top_sentiments) else "rgba(200,120,120,0.6)") for i in range(len(nodes))]
#         ),
#         link=dict(
#             source=source,
#             target=target,
#             value=value
#         )
#     )])
#     fig.update_layout(margin=dict(l=10, r=10, t=20, b=20))
#     return fig

# def ensure_take_n(rule, default=10):
#     if "take_n" not in rule or not isinstance(rule["take_n"], int):
#         rule["take_n"] = default

# def safe_index_choice(current_value, choices):
#     """Return index for selectbox without raising if current_value not in choices."""
#     try:
#         return (["(Any)"] + choices).index(current_value)
#     except Exception:
#         return 0

# # ---------------------------
# # LOAD DATA
# # ---------------------------
# df = load_dataset(DATA_PATH)

# st.set_page_config(page_title="ESG Subset & Sankey", layout="wide")
# st.title("ESG — Data Subset / Sankey / Multi-rules")

# # show the uploaded image (for reference)
# if os.path.exists(UPLOADED_IMAGE_PATH):
#     st.sidebar.image(UPLOADED_IMAGE_PATH, caption="Reference screenshot", use_column_width=True)

# # ---------------------------
# # CHECK COLUMNS
# # ---------------------------
# required_cols = ["aspect_category", "sentiment", "tone"]
# missing = [c for c in required_cols if c not in df.columns]
# if missing:
#     st.error(f"Required columns missing: {missing}")
#     st.stop()

# aspect_options = sorted(df["aspect_category"].dropna().unique().tolist())
# sentiment_options = sorted(df["sentiment"].dropna().unique().tolist())
# tone_options = sorted(df["tone"].dropna().unique().tolist())

# # ---------------------------
# # SIDEBAR: WATERFALL FILTER (moved)
# # ---------------------------
# st.sidebar.header("1️⃣ Waterfall Filter (Step-by-step)")

# selected_aspect = st.sidebar.selectbox("Aspect Category", ["(All)"] + aspect_options, index=0)
# df_step1 = df if selected_aspect == "(All)" else df[df["aspect_category"] == selected_aspect]
# st.sidebar.caption(f"Matching rows after Aspect filter: {len(df_step1):,}")

# sentiment_dynamic = sorted(df_step1["sentiment"].unique().tolist())
# selected_sentiment = st.sidebar.selectbox("Sentiment", ["(All)"] + sentiment_dynamic, index=0 if "(All)" else 0)
# df_step2 = df_step1 if selected_sentiment == "(All)" else df_step1[df_step1["sentiment"] == selected_sentiment]
# st.sidebar.caption(f"Matching rows after Sentiment filter: {len(df_step2):,}")

# tone_dynamic = sorted(df_step2["tone"].unique().tolist())
# selected_tone = st.sidebar.selectbox("Tone", ["(All)"] + tone_dynamic, index=0 if "(All)" else 0)
# df_filtered = df_step2 if selected_tone == "(All)" else df_step2[df_step2["tone"] == selected_tone]
# st.sidebar.success(f"🎯 Final Waterfall Count: {len(df_filtered):,}")

# # ---------------------------
# # Sidebar: Sankey for selected Aspect (D: Aspect -> Sentiment -> Tone)
# # ---------------------------
# st.sidebar.markdown("---")
# st.sidebar.header("Sankey: Selected Aspect → Sentiment → Tone")

# if selected_aspect == "(All)":
#     st.sidebar.info("Select an Aspect to view a Sankey diagram specific to that aspect.")
# else:
#     sankey_for_aspect_df = df[df["aspect_category"] == selected_aspect]
#     sankey_fig = build_sankey_from_df(sankey_for_aspect_df, top_n_aspects=1, top_n_sentiments=20, top_n_tones=20)
#     if sankey_fig:
#         st.sidebar.plotly_chart(sankey_fig, use_container_width=True)
#     else:
#         st.sidebar.warning("No data to build Sankey for this Aspect.")

# # ---------------------------
# # MAIN: Multi-Combination Rules
# # ---------------------------
# st.markdown("## 2️⃣ Build Multiple Filter Combinations")
# if "filter_rows" not in st.session_state:
#     st.session_state.filter_rows = []

# # Add new rule button
# add_col1, add_col2 = st.columns([1, 3])
# with add_col1:
#     if st.button("➕ Add new filter rule"):
#         st.session_state.filter_rows.append({"aspect": "(Any)", "sentiment": "(Any)", "tone": "(Any)", "take_n": 10})

# # Option: choose random sampling vs head
# sample_mode = st.selectbox("Row selection mode for each rule", ["head (first N rows)", "random sample (N rows)"])

# results_summary = []
# selected_rows_combined = []

# # Render rules
# for i, rule in enumerate(st.session_state.filter_rows):
#     ensure_take_n(rule, default=10)

#     with st.expander(f"Rule #{i+1}", expanded=True):
#         c1, c2, c3, c4 = st.columns([2,2,2,1])

#         rule["aspect"] = c1.selectbox(
#             "Aspect",
#             ["(Any)"] + aspect_options,
#             index=safe_index_choice(rule.get("aspect", "(Any)"), aspect_options),
#             key=f"aspect_{i}"
#         )

#         rule["sentiment"] = c2.selectbox(
#             "Sentiment",
#             ["(Any)"] + sentiment_options,
#             index=safe_index_choice(rule.get("sentiment", "(Any)"), sentiment_options),
#             key=f"sentiment_{i}"
#         )

#         rule["tone"] = c3.selectbox(
#             "Tone",
#             ["(Any)"] + tone_options,
#             index=safe_index_choice(rule.get("tone", "(Any)"), tone_options),
#             key=f"tone_{i}"
#         )

#         if c4.button("🗑 Remove", key=f"remove_{i}"):
#             st.session_state.filter_rows.pop(i)
#             st.experimental_rerun()

#         # apply filter
#         temp = df.copy()
#         if rule["aspect"] != "(Any)":
#             temp = temp[temp["aspect_category"] == rule["aspect"]]
#         if rule["sentiment"] != "(Any)":
#             temp = temp[temp["sentiment"] == rule["sentiment"]]
#         if rule["tone"] != "(Any)":
#             temp = temp[temp["tone"] == rule["tone"]]

#         match_count = len(temp)
#         st.info(f"📌 Matches: **{match_count} rows**")

#         # take_n input
#         rule["take_n"] = c1.number_input(
#             "Number of rows to take",
#             min_value=1,
#             max_value=max(1, match_count),
#             value=min(rule.get("take_n", 10), max(1, match_count)),
#             key=f"take_n_{i}"
#         )

#         # mini sankey for this rule (sentiment->tone limited)
#         if match_count > 0:
#             mini_sankey_fig = build_sankey_from_df(temp, top_n_aspects=5, top_n_sentiments=10, top_n_tones=10)
#             if mini_sankey_fig:
#                 st.markdown("**Mini Sankey (filtered rows)**")
#                 st.plotly_chart(mini_sankey_fig, use_container_width=True)

#             # pie chart for selected rows
#             selected_sample = temp.head(rule["take_n"]) if sample_mode.startswith("head") else temp.sample(min(rule["take_n"], len(temp)))
#             pie_counts = selected_sample["sentiment"].value_counts().reset_index()
#             pie_counts.columns = ["label", "value"]
#             pie_fig = go.Figure(data=[go.Pie(labels=pie_counts["label"], values=pie_counts["value"], hole=0.35)])
#             st.markdown("**Pie (sentiment) of selected rows**")
#             st.plotly_chart(pie_fig, use_container_width=True)

#             # save selected data for download / combined output
#             selected_rows_combined.append(selected_sample)

#         results_summary.append({
#             "Rule #": i+1,
#             "Aspect": rule["aspect"],
#             "Sentiment": rule["sentiment"],
#             "Tone": rule["tone"],
#             "Count": match_count,
#             "Take N": rule["take_n"]
#         })

# # Summary table of rules
# if results_summary:
#     st.markdown("### 📊 Summary of Rules")
#     st.dataframe(pd.DataFrame(results_summary), use_container_width=True)

#     # Show detailed data per rule
#     st.markdown("### 📌 Selected Data (per rule)")
#     for idx, rule in enumerate(st.session_state.filter_rows):
#         # re-apply to get head/random sample that user requested
#         temp = df.copy()
#         if rule["aspect"] != "(Any)":
#             temp = temp[temp["aspect_category"] == rule["aspect"]]
#         if rule["sentiment"] != "(Any)":
#             temp = temp[temp["sentiment"] == rule["sentiment"]]
#         if rule["tone"] != "(Any)":
#             temp = temp[temp["tone"] == rule["tone"]]

#         if len(temp) == 0:
#             st.write(f"Rule #{idx+1}: no matches")
#             continue

#         sel = temp.head(rule["take_n"]) if sample_mode.startswith("head") else temp.sample(min(rule["take_n"], len(temp)))
#         st.markdown(f"#### Rule #{idx+1} — Showing {len(sel)} rows")
#         st.dataframe(sel, use_container_width=True)

#     # Combined download of all selected rows
#     if selected_rows_combined:
#         combined_df = pd.concat(selected_rows_combined).drop_duplicates().reset_index(drop=True)
#         csv_bytes = combined_df.to_csv(index=False).encode("utf-8")
#         st.download_button("⬇ Download combined selected rows (CSV)", data=csv_bytes, file_name="selected_rows.csv", mime="text/csv")

# # ---------------------------
# # Footer / notes
# # ---------------------------
# st.markdown("---")
# st.caption("Sankey nodes limited to top-N items for clarity. Adjust top_n ranks in build_sankey_from_df if you want more granularity.")

# 4_Data_Subset_final.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys
from io import BytesIO

# ---------------------------
# MUST RUN FIRST (Streamlit requirement)
# ---------------------------
st.set_page_config(page_title="ESG Subset & Sankey", layout="wide")

# ---------------------------
# PATHS / CONSTANTS
# ---------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "output_in_csv.csv")

# Developer-provided image path from conversation history (use as reference)
UPLOADED_IMAGE_PATH = "/mnt/data/Screenshot 2025-11-21 at 10.57.18.png"

# ---------------------------
# HELPERS
# ---------------------------
@st.cache_data
def load_dataset(path=DATA_PATH):
    df_local = pd.read_csv(path)
    df_local.columns = df_local.columns.str.lower().str.strip()
    return df_local

def build_sankey_from_df(df_in, top_n_aspects=20, top_n_sentiments=10, top_n_tones=10):
    """
    Build Sankey nodes & links for Aspect -> Sentiment -> Tone.
    Keeps only top-N nodes per layer to reduce clutter.
    Returns a plotly Figure or None if insufficient data.
    """
    if df_in is None or df_in.empty:
        return None

    # Frequencies
    aspect_freq = df_in["aspect_category"].value_counts()
    sentiment_freq = df_in["sentiment"].value_counts()
    tone_freq = df_in["tone"].value_counts()

    top_aspects = aspect_freq.head(top_n_aspects).index.tolist()
    top_sentiments = sentiment_freq.head(top_n_sentiments).index.tolist()
    top_tones = tone_freq.head(top_n_tones).index.tolist()

    # Keep only rows that include nodes of interest (some flexibility)
    df_filtered = df_in[
        df_in["aspect_category"].isin(top_aspects) |
        df_in["sentiment"].isin(top_sentiments) |
        df_in["tone"].isin(top_tones)
    ]

    if df_filtered.empty:
        return None

    # Node ordering: aspects, sentiments, tones
    nodes = top_aspects + [s for s in top_sentiments if s not in top_aspects] + [t for t in top_tones if t not in (top_aspects + top_sentiments)]
    node_index = {n: i for i, n in enumerate(nodes)}

    # Links
    a_to_s = df_filtered.groupby(["aspect_category", "sentiment"]).size().reset_index(name="value")
    s_to_t = df_filtered.groupby(["sentiment", "tone"]).size().reset_index(name="value")

    source = []
    target = []
    value = []

    for _, r in a_to_s.iterrows():
        a, s, v = r["aspect_category"], r["sentiment"], int(r["value"])
        if a in node_index and s in node_index:
            source.append(node_index[a]); target.append(node_index[s]); value.append(v)

    for _, r in s_to_t.iterrows():
        s, t, v = r["sentiment"], r["tone"], int(r["value"])
        if s in node_index and t in node_index:
            source.append(node_index[s]); target.append(node_index[t]); value.append(v)

    if not nodes or not source:
        return None

    # Colors per layer for readability
    colors = []
    n_as = len(top_aspects)
    n_sent = len([s for s in top_sentiments if s not in top_aspects])
    n_tone = len([t for t in top_tones if t not in (top_aspects + top_sentiments)])
    for i in range(len(nodes)):
        if i < n_as:
            colors.append("rgba(55,126,184,0.7)")
        elif i < n_as + n_sent:
            colors.append("rgba(77,175,74,0.7)")
        else:
            colors.append("rgba(228,26,28,0.7)")

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(label=nodes, pad=15, thickness=18, color=colors),
        link=dict(source=source, target=target, value=value)
    )])
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=480)
    return fig

def ensure_take_n(rule, default=10):
    if "take_n" not in rule or not isinstance(rule["take_n"], int):
        rule["take_n"] = default

def safe_index_choice(current_value, choices):
    try:
        return (["(Any)"] + choices).index(current_value)
    except Exception:
        return 0

def df_to_csv_bytes(df_obj):
    return df_obj.to_csv(index=False).encode("utf-8")

# ---------------------------
# LOAD DATA
# ---------------------------
df = load_dataset(DATA_PATH)

# Basic validations
required_cols = ["aspect_category", "sentiment", "tone"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Required columns missing: {missing}")
    st.stop()

aspect_options = sorted(df["aspect_category"].dropna().unique().tolist())
sentiment_options = sorted(df["sentiment"].dropna().unique().tolist())
tone_options = sorted(df["tone"].dropna().unique().tolist())

# ---------------------------
# SIDEBAR: Waterfall filter (moved)
# ---------------------------
st.sidebar.header("1️⃣ Waterfall Filter (Step-by-step)")

selected_aspect = st.sidebar.selectbox("Aspect Category", ["(All)"] + aspect_options, index=0)
df_step1 = df if selected_aspect == "(All)" else df[df["aspect_category"] == selected_aspect]
st.sidebar.caption(f"Matching rows after Aspect filter: {len(df_step1):,}")

sentiment_dynamic = sorted(df_step1["sentiment"].dropna().unique().tolist())
selected_sentiment = st.sidebar.selectbox("Sentiment", ["(All)"] + sentiment_dynamic, index=0)
df_step2 = df_step1 if selected_sentiment == "(All)" else df_step1[df_step1["sentiment"] == selected_sentiment]
st.sidebar.caption(f"Matching rows after Sentiment filter: {len(df_step2):,}")

tone_dynamic = sorted(df_step2["tone"].dropna().unique().tolist())
selected_tone = st.sidebar.selectbox("Tone", ["(All)"] + tone_dynamic, index=0)
df_filtered = df_step2 if selected_tone == "(All)" else df_step2[df_step2["tone"] == selected_tone]
st.sidebar.success(f"🎯 Final Waterfall Count: {len(df_filtered):,}")

# Show reference image (if exists)
if os.path.exists(UPLOADED_IMAGE_PATH):
    st.sidebar.markdown("---")
    st.sidebar.image(UPLOADED_IMAGE_PATH, caption="Reference screenshot", use_column_width=True)

# ---------------------------------------------------------
# Define sliders FIRST (even if visually below)
# ---------------------------------------------------------
top_n_aspects = st.sidebar.slider("Top N aspects (sankey)", 3, 50, 20, key="topA")
top_n_sent = st.sidebar.slider("Top N sentiments (sankey)", 3, 30, 12, key="topS")
top_n_tone = st.sidebar.slider("Top N tones (sankey)", 3, 30, 12, key="topT")

# ---------------------------------------------------------
# Put Sankey ABOVE sliders using a container
# ---------------------------------------------------------
cond_sankey_container = st.sidebar.container()

cond_sankey_container.markdown("---")
cond_sankey_container.header("Conditional Sankey (selected aspect)")

if selected_aspect == "(All)":
    cond_sankey_container.info("Select an Aspect to view the Sankey.")
else:
    subset_df = df[df["aspect_category"] == selected_aspect]
    sankey_fig = build_sankey_from_df(
        subset_df,
        top_n_aspects=1,
        top_n_sentiments=top_n_sent,
        top_n_tones=top_n_tone
    )
    if sankey_fig:
        cond_sankey_container.plotly_chart(sankey_fig, use_container_width=True)
    else:
        cond_sankey_container.warning("No data to build Sankey for this Aspect.")

# ---------------------------------------------------------
# NOW visually place sliders below Sankey
# ---------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("Sankey Controls")
st.sidebar.caption("Adjust the resolution of the Sankey above.")


# # Sankey top-N controls in sidebar
# st.sidebar.markdown("---")
# st.sidebar.header("Sankey controls")
# top_n_aspects = st.sidebar.slider("Top N aspects (sankey)", min_value=3, max_value=50, value=20, step=1)
# top_n_sent = st.sidebar.slider("Top N sentiments (sankey)", min_value=3, max_value=30, value=12, step=1)
# top_n_tone = st.sidebar.slider("Top N tones (sankey)", min_value=3, max_value=30, value=12, step=1)

# # Display conditional sankey for selected aspect (Aspect->Sentiment->Tone)
# st.sidebar.markdown("---")
# st.sidebar.header("Conditional Sankey (selected aspect)")

# if selected_aspect == "(All)":
#     st.sidebar.info("Select an Aspect to view the Aspect → Sentiment → Tone Sankey.")
# else:
#     sankey_for_aspect_df = df[df["aspect_category"] == selected_aspect]
#     sankey_fig = build_sankey_from_df(sankey_for_aspect_df, top_n_aspects=1, top_n_sentiments=top_n_sent, top_n_tones=top_n_tone)
#     if sankey_fig:
#         st.sidebar.plotly_chart(sankey_fig, use_container_width=True)
#     else:
#         st.sidebar.warning("No data available to build Sankey for the selected aspect.")

# ---------------------------
# MAIN: Page header & Global sankey
# ---------------------------
st.title("ESG — Data Subset & Sankey")
st.markdown("Use the sidebar to filter (Waterfall) and control Sankey display. Build multi-rule combinations below.")

st.markdown("## Global Sankey (Aspect → Sentiment → Tone, top-N per layer)")
global_sankey = build_sankey_from_df(df, top_n_aspects=top_n_aspects, top_n_sentiments=top_n_sent, top_n_tones=top_n_tone)
if global_sankey:
    st.plotly_chart(global_sankey, use_container_width=True)
else:
    st.info("Not enough data to build global Sankey.")

# ---------------------------
# Multi-combination rules
# ---------------------------
st.markdown("---")
st.header("2️⃣ Build Multiple Filter Combinations")

if "filter_rows" not in st.session_state:
    st.session_state.filter_rows = []

# Add new rule
col_add, col_mode = st.columns([1, 3])
with col_add:
    if st.button("➕ Add new filter rule"):
        st.session_state.filter_rows.append({"aspect": "(Any)", "sentiment": "(Any)", "tone": "(Any)", "take_n": 10})

sample_mode = col_mode.selectbox("Row selection mode for each rule", ["head (first N rows)", "random sample (N rows)"])

results_summary = []
selected_rows_combined = []

for i, rule in enumerate(st.session_state.filter_rows):
    ensure_take_n(rule, default=10)

    with st.expander(f"Rule #{i+1}", expanded=True):
        # safe selectboxes with previous values (no crash if older rules miss keys)
        c1, c2, c3, c4 = st.columns([2,2,2,1])

        rule["aspect"] = c1.selectbox(
            "Aspect",
            ["(Any)"] + aspect_options,
            index=safe_index_choice(rule.get("aspect", "(Any)"), aspect_options),
            key=f"aspect_{i}"
        )

        rule["sentiment"] = c2.selectbox(
            "Sentiment",
            ["(Any)"] + sentiment_options,
            index=safe_index_choice(rule.get("sentiment", "(Any)"), sentiment_options),
            key=f"sentiment_{i}"
        )

        rule["tone"] = c3.selectbox(
            "Tone",
            ["(Any)"] + tone_options,
            index=safe_index_choice(rule.get("tone", "(Any)"), tone_options),
            key=f"tone_{i}"
        )

        if c4.button("🗑 Remove", key=f"remove_{i}"):
            st.session_state.filter_rows.pop(i)
            st.experimental_rerun()

        # apply filter
        temp = df.copy()
        if rule["aspect"] != "(Any)":
            temp = temp[temp["aspect_category"] == rule["aspect"]]
        if rule["sentiment"] != "(Any)":
            temp = temp[temp["sentiment"] == rule["sentiment"]]
        if rule["tone"] != "(Any)":
            temp = temp[temp["tone"] == rule["tone"]]

        match_count = len(temp)
        st.info(f"📌 Matches: **{match_count} rows**")

        # number input to choose how many rows to take
        rule["take_n"] = c1.number_input(
            "Number of rows to take",
            min_value=1,
            max_value=max(1, match_count),
            value=min(rule.get("take_n", 10), max(1, match_count)),
            key=f"take_n_{i}"
        )

        # mini sankey for this filtered subset
        if match_count > 0:
            st.markdown("**Mini Sankey (filtered subset)**")
            mini_sankey_fig = build_sankey_from_df(temp, top_n_aspects=3, top_n_sentiments=8, top_n_tones=8)
            if mini_sankey_fig:
                st.plotly_chart(mini_sankey_fig, use_container_width=True)

            # selected rows by head or random
            sel_rows = temp.head(rule["take_n"]) if sample_mode.startswith("head") else temp.sample(min(rule["take_n"], len(temp)))
            selected_rows_combined.append(sel_rows)

            # pie chart (sentiment distribution) for chosen rows
            pie_counts = sel_rows["sentiment"].value_counts().reset_index()
            pie_counts.columns = ["label", "value"]
            pie_fig = go.Figure(data=[go.Pie(labels=pie_counts["label"], values=pie_counts["value"], hole=0.4)])
            st.markdown("**Pie: Sentiment distribution of selected rows**")
            st.plotly_chart(pie_fig, use_container_width=True)

            # per-rule download button
            csv_bytes = df_to_csv_bytes(sel_rows)
            st.download_button(
                label=f"⬇ Download Rule #{i+1} selected rows (CSV)",
                data=csv_bytes,
                file_name=f"rule_{i+1}_selected_rows.csv",
                mime="text/csv"
            )

        results_summary.append({
            "Rule #": i+1,
            "Aspect": rule["aspect"],
            "Sentiment": rule["sentiment"],
            "Tone": rule["tone"],
            "Count": match_count,
            "Take N": rule["take_n"]
        })

# Display summary / details
if results_summary:
    st.markdown("### 📊 Summary of Rules")
    st.dataframe(pd.DataFrame(results_summary), use_container_width=True)

    st.markdown("### 📌 Detailed Selected Rows per Rule")
    for idx, rule in enumerate(st.session_state.filter_rows):
        temp = df.copy()
        if rule["aspect"] != "(Any)":
            temp = temp[temp["aspect_category"] == rule["aspect"]]
        if rule["sentiment"] != "(Any)":
            temp = temp[temp["sentiment"] == rule["sentiment"]]
        if rule["tone"] != "(Any)":
            temp = temp[temp["tone"] == rule["tone"]]

        if temp.empty:
            st.write(f"Rule #{idx+1}: no matches")
            continue

        sel = temp.head(rule["take_n"]) if sample_mode.startswith("head") else temp.sample(min(rule["take_n"], len(temp)))
        st.markdown(f"#### Rule #{idx+1} — Showing {len(sel)} rows")
        st.dataframe(sel, use_container_width=True)

    # Combined download of all selected rows
    if selected_rows_combined:
        combined_df = pd.concat(selected_rows_combined).drop_duplicates().reset_index(drop=True)
        st.markdown("### ⤓ Download combined selected rows")
        csv_bytes_all = df_to_csv_bytes(combined_df)
        st.download_button("⬇ Download combined selected rows (CSV)", data=csv_bytes_all, file_name="combined_selected_rows.csv", mime="text/csv")

st.markdown("---")
st.caption("Notes: Sankey nodes are limited to top-N items per layer (controls in sidebar). Use random sampling if you want non-deterministic picks.")
