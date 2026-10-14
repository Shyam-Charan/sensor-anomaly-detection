"""Download the series and labels from the Numenta Anomaly Benchmark (NAB) into data/."""
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/numenta/NAB/master"
FILES = {
    "data/realKnownCause/machine_temperature_system_failure.csv": "machine_temperature_system_failure.csv",
    "labels/combined_windows.json": "combined_windows.json",
}
out = Path(__file__).resolve().parents[1] / "data"
out.mkdir(exist_ok=True)
for remote, local in FILES.items():
    urllib.request.urlretrieve(f"{BASE}/{remote}", out / local)
    print("downloaded", local)
