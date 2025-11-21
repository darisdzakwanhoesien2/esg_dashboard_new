# import streamlit as st
# import pandas as pd
# import os
# import sys

# # --------------------------------------------------------------
# # LOAD DATA (same method as main app)
# # --------------------------------------------------------------

# # Fix path
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
# sys.path.append(PROJECT_ROOT)

# @st.cache_data
# def load_dataset():
#     data_path = os.path.join(PROJECT_ROOT, "data/output_in_csv.csv")
#     df = pd.read_csv(data_path)
#     df.columns = df.columns.str.lower().str.strip()
#     return df

# df = load_dataset()

# st.header("🔎 ESG Waterfall Filter — Aspect, Sentiment, Tone")

# # --------------------------------------------------------------
# # CHECK REQUIRED COLUMNS
# # --------------------------------------------------------------
# required_cols = ["aspect_category", "sentiment", "tone"]
# missing = [c for c in required_cols if c not in df.columns]

# if missing:
#     st.error(f"❌ Missing required columns: {missing}")
#     st.stop()

# # Preload options
# aspect_options = sorted(df["aspect_category"].dropna().unique())
# sentiment_options = sorted(df["sentiment"].dropna().unique())
# tone_options = sorted(df["tone"].dropna().unique())

# # ==============================================================
# # 1) WATERFALL FILTER
# # ==============================================================
# st.subheader("1️⃣ Waterfall Filter (Step-by-step filtering)")

# # Step 1 — Aspect
# selected_aspect = st.selectbox("Aspect Category", ["(All)"] + aspect_options)

# df_step1 = df if selected_aspect == "(All)" else df[df["aspect_category"] == selected_aspect]
# st.caption(f"Matching rows after Aspect filter: **{len(df_step1)}**")

# # Step 2 — Sentiment
# sentiment_dynamic_options = sorted(df_step1["sentiment"].unique())
# selected_sentiment = st.selectbox("Sentiment", ["(All)"] + sentiment_dynamic_options)

# df_step2 = df_step1 if selected_sentiment == "(All)" else df_step1[df_step1["sentiment"] == selected_sentiment]
# st.caption(f"Matching rows after Sentiment filter: **{len(df_step2)}**")

# # Step 3 — Tone
# tone_dynamic_options = sorted(df_step2["tone"].unique())
# selected_tone = st.selectbox("Tone", ["(All)"] + tone_dynamic_options)

# df_filtered = df_step2 if selected_tone == "(All)" else df_step2[df_step2["tone"] == selected_tone]

# st.success(f"🎯 Final result count: **{len(df_filtered)}**")
# st.dataframe(df_filtered, use_container_width=True)

# # ==============================================================
# # 2) MULTI-COMBINATION FILTER
# # ==============================================================
# st.markdown("---")
# st.subheader("2️⃣ Build Multiple Filter Combinations")

# if "filter_rows" not in st.session_state:
#     st.session_state.filter_rows = []

# # Add new rule
# if st.button("➕ Add new filter rule"):
#     st.session_state.filter_rows.append({
#         "aspect": "(Any)",
#         "sentiment": "(Any)",
#         "tone": "(Any)"
#     })

# # Render rules + collect results
# all_rule_results = []

# for i, rule in enumerate(st.session_state.filter_rows):

#     with st.expander(f"Rule #{i+1}", expanded=True):

#         c1, c2, c3, c4 = st.columns([2,2,2,1])

#         rule["aspect"] = c1.selectbox(
#             "Aspect",
#             ["(Any)"] + aspect_options,
#             index=(["(Any)"] + aspect_options).index(rule["aspect"]),
#             key=f"aspect_{i}"
#         )

#         rule["sentiment"] = c2.selectbox(
#             "Sentiment",
#             ["(Any)"] + sentiment_options,
#             index=(["(Any)"] + sentiment_options).index(rule["sentiment"]),
#             key=f"sentiment_{i}"
#         )

#         rule["tone"] = c3.selectbox(
#             "Tone",
#             ["(Any)"] + tone_options,
#             index=(["(Any)"] + tone_options).index(rule["tone"]),
#             key=f"tone_{i}"
#         )

#         # Remove rule
#         if c4.button("🗑", key=f"remove_{i}"):
#             st.session_state.filter_rows.pop(i)
#             st.experimental_rerun()

# # --------------------------------------------------------------
# # APPLY MULTI-RULE FILTERS
# # --------------------------------------------------------------

# for idx, rule in enumerate(st.session_state.filter_rows):

#     temp = df.copy()

#     if rule["aspect"] != "(Any)":
#         temp = temp[temp["aspect_category"] == rule["aspect"]]
#     if rule["sentiment"] != "(Any)":
#         temp = temp[temp["sentiment"] == rule["sentiment"]]
#     if rule["tone"] != "(Any)":
#         temp = temp[temp["tone"] == rule["tone"]]

#     all_rule_results.append({
#         "Rule #": idx+1,
#         "Aspect": rule["aspect"],
#         "Sentiment": rule["sentiment"],
#         "Tone": rule["tone"],
#         "Count": len(temp),
#         "Data": temp
#     })

# # --------------------------------------------------------------
# # DISPLAY SUMMARY
# # --------------------------------------------------------------
# if all_rule_results:

#     st.markdown("### 📊 Summary of All Rules")

#     summary_df = pd.DataFrame([{
#         "Rule #": r["Rule #"],
#         "Aspect": r["Aspect"],
#         "Sentiment": r["Sentiment"],
#         "Tone": r["Tone"],
#         "Count": r["Count"]
#     } for r in all_rule_results])

#     st.dataframe(summary_df, use_container_width=True)

#     st.markdown("### 📌 Detailed Outputs for Each Rule")
#     for r in all_rule_results:
#         st.markdown(f"#### Rule #{r['Rule #']}")
#         st.dataframe(r["Data"], use_container_width=True)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys

# --------------------------------------------------------------
# LOAD DATA (consistent with your main app)
# --------------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

@st.cache_data
def load_dataset():
    data_path = os.path.join(PROJECT_ROOT, "data/output_in_csv.csv")
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower().str.strip()
    return df

df = load_dataset()

st.header("🔎 ESG Waterfall & Multi-Combination Filter")

# --------------------------------------------------------------
# CHECK REQUIRED COLUMN NAMES
# --------------------------------------------------------------
required_cols = ["aspect_category", "sentiment", "tone"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"❌ Missing required columns: {missing}")
    st.stop()

aspect_options = sorted(df["aspect_category"].dropna().unique())
sentiment_options = sorted(df["sentiment"].dropna().unique())
tone_options = sorted(df["tone"].dropna().unique())

# ==============================================================
# 🔷 0) SANKEY DIAGRAM
# ==============================================================
# st.markdown("## 🪢 Flow Diagram — Aspect → Sentiment → Tone")

# sankey_df = df.groupby(["aspect_category", "sentiment", "tone"]) \
#               .size().reset_index(name="count")

# all_aspects = sankey_df["aspect_category"].unique().tolist()
# all_sentiments = sankey_df["sentiment"].unique().tolist()
# all_tones = sankey_df["tone"].unique().tolist()

# nodes = all_aspects + all_sentiments + all_tones
# node_index = {n: i for i, n in enumerate(nodes)}

# links = {"source": [], "target": [], "value": []}

# # Aspect → Sentiment
# for _, row in sankey_df.iterrows():
#     links["source"].append(node_index[row["aspect_category"]])
#     links["target"].append(node_index[row["sentiment"]])
#     links["value"].append(row["count"])

# # Sentiment → Tone
# for _, row in sankey_df.iterrows():
#     links["source"].append(node_index[row["sentiment"]])
#     links["target"].append(node_index[row["tone"]])
#     links["value"].append(row["count"])

# fig = go.Figure(data=[go.Sankey(
#     node=dict(
#         pad=15,
#         thickness=18,
#         label=nodes,
#         color="rgba(100,100,200,0.4)"
#     ),
#     link=links
# )])

# st.plotly_chart(fig, use_container_width=True)

# ==============================================================
# 0) SANKEY DIAGRAM (GLOBAL, SORTED)
# ==============================================================

st.markdown("## 🪢 Sankey: Aspect → Sentiment → Tone (Sorted by Frequency)")

sankey_df = df.groupby(["aspect_category", "sentiment", "tone"]).size().reset_index(name="count")

# Sort nodes by count (descending)
aspect_freq = sankey_df.groupby("aspect_category")["count"].sum().sort_values(ascending=False)
sentiment_freq = sankey_df.groupby("sentiment")["count"].sum().sort_values(ascending=False)
tone_freq = sankey_df.groupby("tone")["count"].sum().sort_values(ascending=False)

all_aspects = aspect_freq.index.tolist()
all_sentiments = sentiment_freq.index.tolist()
all_tones = tone_freq.index.tolist()

nodes = all_aspects + all_sentiments + all_tones
node_index = {n: i for i, n in enumerate(nodes)}

links = {"source": [], "target": [], "value": []}

for _, row in sankey_df.iterrows():
    a, s, t, c = row["aspect_category"], row["sentiment"], row["tone"], row["count"]
    links["source"].append(node_index[a])
    links["target"].append(node_index[s])
    links["value"].append(c)
    links["source"].append(node_index[s])
    links["target"].append(node_index[t])
    links["value"].append(c)

fig = go.Figure(data=[go.Sankey(
    arrangement="snap",
    node=dict(
        pad=18,
        thickness=18,
        label=nodes,
        color="rgba(40,120,200,0.35)"
    ),
    link=links
)])

st.plotly_chart(fig, use_container_width=True)
st.markdown("---")

# ==============================================================
# 1️⃣ WATERFALL FILTER (Step-by-Step)
# ==============================================================
st.markdown("---")
st.subheader("1️⃣ Waterfall Filter (Step-by-Step)")

# Step 1 — Aspect
selected_aspect = st.selectbox("Aspect Category", ["(All)"] + aspect_options)
df_step1 = df if selected_aspect == "(All)" else df[df["aspect_category"] == selected_aspect]
st.caption(f"Matching rows after Aspect filter: **{len(df_step1)}**")

# Step 2 — Sentiment
sent_opts = sorted(df_step1["sentiment"].unique())
selected_sentiment = st.selectbox("Sentiment", ["(All)"] + sent_opts)
df_step2 = df_step1 if selected_sentiment == "(All)" else df_step1[df_step1["sentiment"] == selected_sentiment]
st.caption(f"Matching rows after Sentiment filter: **{len(df_step2)}**")

# Step 3 — Tone
tone_opts = sorted(df_step2["tone"].unique())
selected_tone = st.selectbox("Tone", ["(All)"] + tone_opts)
df_filtered = df_step2 if selected_tone == "(All)" else df_step2[df_step2["tone"] == selected_tone]

st.success(f"🎯 Final Waterfall Count: **{len(df_filtered)}**")
st.dataframe(df_filtered, use_container_width=True)


# ==============================================================
# 2️⃣ MULTI-COMBINATION FILTER
# ==============================================================
# st.markdown("---")
# st.subheader("2️⃣ Build Multiple Filter Combinations")

# # Store rules
# if "filter_rows" not in st.session_state:
#     st.session_state.filter_rows = []

# # Add new rule
# if st.button("➕ Add new filter rule"):
#     st.session_state.filter_rows.append({
#         "aspect": "(Any)",
#         "sentiment": "(Any)",
#         "tone": "(Any)",
#         "take_n": 10
#     })

# results_summary = []

# # RENDER EACH RULE
# for i, rule in enumerate(st.session_state.filter_rows):

#     with st.expander(f"Rule #{i+1}", expanded=True):

#         c1, c2, c3, c4 = st.columns([2,2,2,1])

#         # Dropdowns for rule conditions
#         rule["aspect"] = c1.selectbox(
#             "Aspect",
#             ["(Any)"] + aspect_options,
#             index=(["(Any)"] + aspect_options).index(rule["aspect"]),
#             key=f"aspect_{i}"
#         )

#         rule["sentiment"] = c2.selectbox(
#             "Sentiment",
#             ["(Any)"] + sentiment_options,
#             index=(["(Any)"] + sentiment_options).index(rule["sentiment"]),
#             key=f"sentiment_{i}"
#         )

#         rule["tone"] = c3.selectbox(
#             "Tone",
#             ["(Any)"] + tone_options,
#             index=(["(Any)"] + tone_options).index(rule["tone"]),
#             key=f"tone_{i}"
#         )

#         # Remove button
#         if c4.button("🗑", key=f"remove_{i}"):
#             st.session_state.filter_rows.pop(i)
#             st.experimental_rerun()

#         # ---- APPLY FILTER ----
#         temp = df.copy()
#         if rule["aspect"] != "(Any)":
#             temp = temp[temp["aspect_category"] == rule["aspect"]]
#         if rule["sentiment"] != "(Any)":
#             temp = temp[temp["sentiment"] == rule["sentiment"]]
#         if rule["tone"] != "(Any)":
#             temp = temp[temp["tone"] == rule["tone"]]

#         match_count = len(temp)
#         st.info(f"📌 Matches: **{match_count} rows**")

#         # ---- SELECT HOW MANY ROWS TO RETURN ----
#         rule["take_n"] = st.number_input(
#             "Number of rows to take",
#             min_value=1,
#             max_value=max(1, match_count),
#             value=min(rule["take_n"], max(1, match_count)),
#             key=f"take_n_{i}"
#         )

#         # Save summary
#         results_summary.append({
#             "Rule #": i+1,
#             "Aspect": rule["aspect"],
#             "Sentiment": rule["sentiment"],
#             "Tone": rule["tone"],
#             "Count": match_count,
#             "Take N": rule["take_n"],
#             "Data": temp.head(rule["take_n"])
#         })

# ==============================================================
# 2) MULTI-COMBINATION FILTER
# ==============================================================
st.subheader("2️⃣ Build Multiple Filter Combinations")

# Initialize session state
if "filter_rows" not in st.session_state:
    st.session_state.filter_rows = []

# Add new rule
if st.button("➕ Add new filter rule"):
    st.session_state.filter_rows.append({
        "aspect": "(Any)",
        "sentiment": "(Any)",
        "tone": "(Any)",
        "take_n": 10
    })

results_summary = []

# Loop through user rules
for i, rule in enumerate(st.session_state.filter_rows):

    # Fix missing take_n (prevents KeyError)
    if "take_n" not in rule:
        rule["take_n"] = 10

    with st.expander(f"Rule #{i+1}", expanded=True):

        # =============== MINI SANKEY ABOVE DROPDOWNS ===============
        st.markdown("##### Mini Sankey for This Rule")

        temp_all = df.copy()
        fig_small = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(
                pad=10,
                thickness=14,
                label=nodes,
                color="rgba(200,200,200,0.15)"
            ),
            link=links
        )])
        st.plotly_chart(fig_small, use_container_width=True)

        # ================= DROPDOWNS =================
        c1, c2, c3, c4 = st.columns([2,2,2,1])

        rule["aspect"] = c1.selectbox(
            "Aspect",
            ["(Any)"] + aspect_options,
            index=(["(Any)"] + aspect_options).index(rule["aspect"]),
            key=f"aspect_{i}"
        )

        rule["sentiment"] = c2.selectbox(
            "Sentiment",
            ["(Any)"] + sentiment_options,
            index=(["(Any)"] + sentiment_options).index(rule["sentiment"]),
            key=f"sentiment_{i}"
        )

        rule["tone"] = c3.selectbox(
            "Tone",
            ["(Any)"] + tone_options,
            index=(["(Any)"] + tone_options).index(rule["tone"]),
            key=f"tone_{i}"
        )

        # Remove rule
        if c4.button("🗑", key=f"remove_{i}"):
            st.session_state.filter_rows.pop(i)
            st.experimental_rerun()

        # ================= APPLY FILTER =================
        temp = df.copy()
        if rule["aspect"] != "(Any)":
            temp = temp[temp["aspect_category"] == rule["aspect"]]
        if rule["sentiment"] != "(Any)":
            temp = temp[temp["sentiment"] == rule["sentiment"]]
        if rule["tone"] != "(Any)":
            temp = temp[temp["tone"] == rule["tone"]]

        match_count = len(temp)

        st.info(f"📌 Matches: **{match_count} rows**")

        # ================= SELECT TAKE_N =================
        rule["take_n"] = st.number_input(
            "Number of rows to take",
            min_value=1,
            max_value=max(1, match_count),
            value=min(rule["take_n"], max(1, match_count)),
            key=f"take_n_{i}"
        )

        # ================= PIE CHART OF SELECTED ============
        if match_count > 0:
            st.markdown("##### 📊 Pie Chart for Selected Subset")

            pie_df = temp.head(rule["take_n"])
            pie_counts = pie_df["tone"].value_counts().reset_index()
            pie_counts.columns = ["tone", "count"]

            pie_fig = go.Figure(data=[go.Pie(
                labels=pie_counts["tone"],
                values=pie_counts["count"],
                hole=0.4
            )])

            st.plotly_chart(pie_fig, use_container_width=True)

        # Save rule summary
        results_summary.append({
            "Rule #": i+1,
            "Aspect": rule["aspect"],
            "Sentiment": rule["sentiment"],
            "Tone": rule["tone"],
            "Count": match_count,
            "Take N": rule["take_n"],
            "Data": temp.head(rule["take_n"])
        })

# SUMMARY TABLE
if results_summary:
    st.markdown("### 📊 Summary of Rules")

    summary_df = pd.DataFrame([{
        "Rule #": r["Rule #"],
        "Aspect": r["Aspect"],
        "Sentiment": r["Sentiment"],
        "Tone": r["Tone"],
        "Count": r["Count"],
        "Take N": r["Take N"]
    } for r in results_summary])

    st.dataframe(summary_df, use_container_width=True)

    st.markdown("### 📌 Full Data for Each Rule")
    for r in results_summary:
        st.markdown(f"#### Rule #{r['Rule #']}")
        st.dataframe(r["Data"], use_container_width=True)

# ==============================================================
# SUMMARY TABLE
# ==============================================================
if results_summary:
    st.markdown("### 📊 Summary of All Rules")

    summary_df = pd.DataFrame([{
        "Rule #": r["Rule #"],
        "Aspect": r["Aspect"],
        "Sentiment": r["Sentiment"],
        "Tone": r["Tone"],
        "Count": r["Count"],
        "Take N": r["Take N"]
    } for r in results_summary])

    st.dataframe(summary_df, use_container_width=True)

    st.markdown("### 📌 Detailed Outputs for Each Rule")
    for r in results_summary:
        st.markdown(f"#### Rule #{r['Rule #']}")
        st.dataframe(r["Data"], use_container_width=True)
