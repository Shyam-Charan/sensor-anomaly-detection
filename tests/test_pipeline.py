import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import label_points
from features import build_features
from evaluate import time_split, windows_detected
from models import ZScoreBaseline
import db


def _series(n=300, seed=0):
    idx = pd.date_range("2014-01-01", periods=n, freq="5min")
    return pd.DataFrame({"value": np.random.default_rng(seed).normal(80, 1, n)}, index=idx)


def test_features_have_no_lookahead():
    df = _series()
    full = build_features(df)
    truncated = build_features(df.iloc[:200])
    t = truncated.index[-1]
    pd.testing.assert_series_equal(full.loc[t], truncated.loc[t])


def test_label_points_marks_window_inclusive():
    idx = pd.date_range("2014-01-01", periods=10, freq="1h")
    y = label_points(idx, [(idx[2], idx[4])])
    assert y.tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]


def test_time_split_is_chronological():
    df = _series()
    X = build_features(df)
    y = pd.Series(0, index=X.index)
    X_tr, X_te, _, _ = time_split(X, y, 0.5)
    assert X_tr.index.max() < X_te.index.min()


def test_zscore_flags_a_spike():
    df = _series()
    df.iloc[250, 0] += 30
    X = build_features(df)
    pred = pd.Series(ZScoreBaseline().predict(X), index=X.index)
    assert pred[df.index[250]] == 1


def test_windows_detected_counts_hits():
    idx = pd.date_range("2014-01-01", periods=10, freq="1h")
    pred = np.zeros(10, dtype=int)
    pred[3] = 1
    windows = [(idx[2], idx[4]), (idx[7], idx[8])]
    assert windows_detected(idx, pred, windows) == (1, 2)


def test_db_roundtrip(tmp_path):
    df = _series(50)
    X = df.copy()
    y = pd.Series(0, index=df.index)
    path = tmp_path / "r.db"
    results = pd.DataFrame([{"model": "m", "precision": 1.0, "recall": 1.0, "f1": 1.0,
                             "alerts": 2, "windows_detected": 1, "windows_total": 1}])
    db.save(path, X, y, {"m": np.zeros(50, dtype=int)}, results)
    assert db.query(path, "SELECT COUNT(*) AS n FROM predictions")["n"][0] == 50
