import pandas as pd


def align_by_sentence(gt_df, pred_df, sentence_col="sentence"):
    """
    Align ground truth and prediction dataframes by sentence.
    For each sentence, keeps min(count_gt, count_pred) rows.
    """

    gt_df = gt_df.copy()
    pred_df = pred_df.copy()

    gt_df["_row_id"] = gt_df.groupby(sentence_col).cumcount()
    pred_df["_row_id"] = pred_df.groupby(sentence_col).cumcount()

    merged = gt_df.merge(
        pred_df,
        on=[sentence_col, "_row_id"],
        suffixes=("_gt", "_pred"),
        how="inner"
    )

    # Clean up
    merged.drop(columns=["_row_id"], inplace=True)

    return merged
