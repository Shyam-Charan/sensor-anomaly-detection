"""Time-aware split and metrics."""
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def time_split(X: pd.DataFrame, y: pd.Series, train_frac=0.5):
    """Chronological split: train on the earlier part, test on the later part. Never shuffle a time series."""
    cut = int(len(X) * train_frac)
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def point_metrics(y_true, y_pred) -> dict:
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def windows_detected(index, y_pred, windows) -> tuple[int, int]:
    """How many labelled windows inside `index` contain at least one alert."""
    pred = pd.Series(y_pred, index=index)
    inside = [(a, b) for a, b in windows if ((index >= a) & (index <= b)).any()]
    hit = sum(1 for a, b in inside if pred[(index >= a) & (index <= b)].any())
    return hit, len(inside)
