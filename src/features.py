"""Rolling-window feature engineering for a univariate sensor series.

Readings arrive every 5 minutes, so 12 steps = 1 hour and 72 steps = 6 hours.
Every feature at time t uses only data up to and including t (no look-ahead).
"""
import numpy as np
import pandas as pd

WINDOWS = (12, 72)


def build_features(df: pd.DataFrame, windows=WINDOWS) -> pd.DataFrame:
    s = df["value"]
    feats = pd.DataFrame(index=df.index)
    feats["value"] = s
    feats["diff_1"] = s.diff()
    for w in windows:
        roll = s.rolling(w)
        feats[f"mean_{w}"] = roll.mean()
        feats[f"std_{w}"] = roll.std()
        feats[f"min_{w}"] = roll.min()
        feats[f"max_{w}"] = roll.max()
        feats[f"zscore_{w}"] = (s - feats[f"mean_{w}"]) / feats[f"std_{w}"].replace(0, np.nan)
    return feats.dropna()
