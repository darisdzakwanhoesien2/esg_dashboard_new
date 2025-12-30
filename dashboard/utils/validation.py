def validate_columns(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    return missing
