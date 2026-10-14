"""Store predictions and metrics in SQLite so they can be queried with SQL."""
import sqlite3
from pathlib import Path

import pandas as pd


def save(db_path, X_test: pd.DataFrame, y_test: pd.Series, preds: dict, results: pd.DataFrame):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    long = []
    for model, pred in preds.items():
        long.append(pd.DataFrame({"timestamp": X_test.index.astype(str), "model": model,
                                  "value": X_test["value"].to_numpy(),
                                  "label": y_test.to_numpy(), "predicted": pred}))
    with sqlite3.connect(db_path) as con:
        pd.concat(long).to_sql("predictions", con, if_exists="replace", index=False)
        results.to_sql("metrics", con, if_exists="replace", index=False)


def query(db_path, sql: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query(sql, con)
