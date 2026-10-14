"""Load the sensor series and its labelled anomaly windows."""
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SERIES = "machine_temperature_system_failure.csv"
LABEL_KEY = "realKnownCause/machine_temperature_system_failure.csv"


def load_series(path=None) -> pd.DataFrame:
    """Return a DataFrame indexed by timestamp with a single 'value' column."""
    df = pd.read_csv(path or DATA_DIR / SERIES, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    return df


def load_windows(path=None) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Labelled anomaly windows (start, end) from the Numenta Anomaly Benchmark."""
    with open(path or DATA_DIR / "combined_windows.json") as f:
        raw = json.load(f)[LABEL_KEY]
    return [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in raw]


def label_points(index: pd.DatetimeIndex, windows) -> pd.Series:
    """1 if a timestamp falls inside any labelled anomaly window, else 0."""
    y = pd.Series(0, index=index, name="label")
    for start, end in windows:
        y[(index >= start) & (index <= end)] = 1
    return y
