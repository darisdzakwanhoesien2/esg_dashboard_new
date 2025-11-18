# esg_dashboard_new
esg-dashboard/
├─ dashboard/
│  ├─ app.py                 # main Streamlit app (use this)
│  ├─ model_utils.py         # model load / inference helpers (hybrid)
│  ├─ finbert_model.py       # MultiTaskFinBERT model class skeleton
│  ├─ assets/
│  │   └─ ...                # images/static assets if needed
├─ requirements_streamlit.txt   # for Streamlit Cloud (NO torch)
├─ requirements_local.txt       # for local dev (with torch, transformers)
└─ README.md
