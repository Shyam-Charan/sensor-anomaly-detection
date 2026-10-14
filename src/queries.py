"""Run every statement in sql/queries.sql against outputs/results.db and print the results."""
from pathlib import Path

import db

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    sql_text = (ROOT / "sql" / "queries.sql").read_text()
    for block in [b.strip() for b in sql_text.split(";") if "SELECT" in b]:
        title = block.splitlines()[0] if block.startswith("--") else ""
        print(title)
        print(db.query(ROOT / "outputs" / "results.db", block).to_string(index=False), "\n")
