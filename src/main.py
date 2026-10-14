"""Run the full pipeline: load -> features -> split -> train -> evaluate -> SQLite -> chart."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from data import load_series, load_windows, label_points
from features import build_features
from models import all_models
from evaluate import time_split, point_metrics, windows_detected
import db

ROOT = Path(__file__).resolve().parents[1]


def run(db_path=ROOT / "outputs" / "results.db", make_plot=True):
    df = load_series()
    windows = load_windows()
    X = build_features(df)
    y = label_points(X.index, windows)
    X_tr, X_te, y_tr, y_te = time_split(X, y)
    print(f"{len(X)} rows | train {len(X_tr)} ({X_tr.index[0].date()}..{X_tr.index[-1].date()}) "
          f"| test {len(X_te)} ({X_te.index[0].date()}..{X_te.index[-1].date()})")
    print(f"anomalous points: train {y_tr.mean():.1%}, test {y_te.mean():.1%}")

    rows, preds = [], {}
    for model in all_models():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        preds[model.name] = pred
        m = point_metrics(y_te, pred)
        hit, total = windows_detected(X_te.index, pred, windows)
        rows.append({"model": model.name, **m, "alerts": int(pred.sum()),
                     "windows_detected": hit, "windows_total": total})
    results = pd.DataFrame(rows)
    print("\n" + results.round(3).to_string(index=False))

    db.save(db_path, X_te, y_te, preds, results)
    print(f"\nSaved predictions and metrics to {db_path}")
    if make_plot:
        import plot
        plot.make(X_te, y_te, preds, ROOT / "outputs" / "anomalies.png")
    return results


if __name__ == "__main__":
    run()
