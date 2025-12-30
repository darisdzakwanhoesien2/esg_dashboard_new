from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd


def normalize_labels(y_true, y_pred):
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    })

    # Convert everything to string
    df["y_true"] = df["y_true"].astype(str).str.strip()
    df["y_pred"] = df["y_pred"].astype(str).str.strip()

    # Treat these as invalid labels
    INVALID = {"nan", "none", "", "null"}

    mask = (
        ~df["y_true"].str.lower().isin(INVALID)
        & ~df["y_pred"].str.lower().isin(INVALID)
    )

    return df.loc[mask, "y_true"], df.loc[mask, "y_pred"], len(df) - mask.sum()


def compute_metrics(y_true, y_pred, average="weighted"):
    y_true_clean, y_pred_clean, dropped = normalize_labels(y_true, y_pred)

    if len(y_true_clean) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "dropped_rows": dropped,
        }

    acc = accuracy_score(y_true_clean, y_pred_clean)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_clean,
        y_pred_clean,
        average=average,
        zero_division=0
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dropped_rows": dropped,
    }


# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# import pandas as pd


# def compute_metrics(y_true, y_pred, average="weighted"):
#     acc = accuracy_score(y_true, y_pred)

#     precision, recall, f1, _ = precision_recall_fscore_support(
#         y_true,
#         y_pred,
#         average=average,
#         zero_division=0
#     )

#     return {
#         "accuracy": acc,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#     }


# def metrics_to_df(metrics_dict):
#     return pd.DataFrame(
#         metrics_dict,
#         index=["Accuracy", "Precision", "Recall", "F1"]
#     ).T
