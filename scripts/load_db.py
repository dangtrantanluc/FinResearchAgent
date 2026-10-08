"""Load data/processed, reference/ and artifacts/ into Postgres (drops and rebuilds the tables).

    docker compose up -d --wait
    python scripts/load_db.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finresearch.db import load_all  # noqa: E402

if __name__ == "__main__":
    start = time.time()
    counts = load_all(ROOT)
    for table, n in counts.items():
        print(f"{table:16s} {n:>10,}")
    print(f"done in {time.time() - start:.0f}s")
