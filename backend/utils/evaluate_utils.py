from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import pandas as pd

def get_metrics(predictions, df):
    """
    Compute accuracy, weighted F1, precision, recall,
    and confusion matrix for a list of prediction dicts
    and the corresponding cleaned DataFrame.
    """
    y_pred = [int(p['label']) for p in predictions]

    # Detect the true‐label column
    label_col = next((c for c in ['sentiment', 'label', 'class'] if c in df.columns), None)
    if not label_col:
        raise ValueError("CSV must include a label column like 'sentiment', 'label', or 'class'.")

    # If labels are strings, map to ints
    if df[label_col].dtype == object:
        mapping = {v: i for i, v in enumerate(sorted(df[label_col].unique()))}
        df[label_col] = df[label_col].map(mapping)

    y_true = df[label_col].astype(int).tolist()

    # Sanity check
    if len(y_true) != len(y_pred):
        raise ValueError(f"❌ Mismatch: {len(y_true)} ground truth vs {len(y_pred)} predictions.")

    # Compute core metrics
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='weighted')
    prec, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    # Build confusion matrix
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy":          acc,
        "f1_score":          f1,
        "precision":         prec,
        "recall":            rec,
        "confusion_matrix":  cm
    }
