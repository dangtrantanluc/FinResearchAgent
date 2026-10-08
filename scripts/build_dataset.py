"""Download the OCR dataset (if needed) and build data/processed/*.parquet.

    python scripts/build_dataset.py [--raw DIR] [--out DIR] [--workers N]
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finresearch.build import build_dataset, download_raw  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=ROOT / "data/raw/ocr_annual_financials")
    ap.add_argument("--out", default=ROOT / "data/processed")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    raw = download_raw(args.raw)
    graded, selected, panel = build_dataset(raw, args.out, ROOT / "reference/companies.csv", args.workers)

    corp = graded[graded.corporate]
    print(f"files: {len(graded):,} | Circular 200 layout: {len(corp):,}")
    print(f"identity checks passed  balance sheet {corp.bs_ok.mean():.1%} | income {corp.is_ok.mean():.1%} | cash flow {corp.cf_ok.mean():.1%}")
    print(f"ticker-years selected: {len(selected):,} from {selected.ticker.nunique():,} tickers")
    print(f"panel rows: {len(panel):,} | usable: {panel.usable.sum():,} | with target: {panel.has_target.sum():,}")
    print(panel[panel.has_target].groupby("year").size().rename("rows with target").to_frame().T.to_string())
