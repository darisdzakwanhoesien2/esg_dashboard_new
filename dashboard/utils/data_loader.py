import pandas as pd
import json
import re
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "data_output.csv"


def extract_json_block(text):
    if not isinstance(text, str):
        return None
    match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except Exception:
        return None


def normalize_json(obj):
    if obj is None:
        return []
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list):
        out = []
        for x in obj:
            out.extend(normalize_json(x))
        return out
    return []


def parse_esg_json(text):
    raw = extract_json_block(text)
    norm = normalize_json(raw)
    return [
        x for x in norm
        if isinstance(x, dict) and "sentence" in x and "aspect" in x
    ]


def load_and_parse():
    raw_df = pd.read_csv(DATA_PATH)

    raw_df["parsed"] = raw_df["text"].apply(parse_esg_json)
    exploded = raw_df.explode("parsed", ignore_index=True)
    parsed_df = pd.json_normalize(exploded["parsed"])

    meta_cols = [c for c in raw_df.columns if c != "parsed"]
    meta = exploded[meta_cols].reset_index(drop=True)

    return pd.concat([meta, parsed_df], axis=1)
