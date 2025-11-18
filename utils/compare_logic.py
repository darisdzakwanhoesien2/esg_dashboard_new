import pandas as pd

def find_missing(output_df, exported_df):
    """
    Find rows in `output_df` whose (filename, page_number)
    do not appear in `exported_df`.
    """
    merged = output_df.merge(
        exported_df[["filename", "page_number"]],
        on=["filename", "page_number"],
        how="left",
        indicator=True
    )
    return merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
