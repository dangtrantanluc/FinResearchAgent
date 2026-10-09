"""Download annual reports into data/reports/<TICKER>/<TICKER>_<YEAR>.pdf.

    python scripts/fetch_reports.py MWG ELC --years 2021-2025

Source: the copies of exchange filings that Vietstock serves at a fixed address,
static2.vietstock.vn/data/<EXCHANGE>/<YEAR>/BCTN/VN/<TICKER>_Baocaothuongnien_<YEAR>.pdf.
Files that already exist are kept. A company's own investor relations page is the
fallback when a year is missing here (FPT's reports came from fpt.com).
"""
import argparse
import csv
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://static2.vietstock.vn/data/{exchange}/{year}/BCTN/VN/{ticker}_Baocaothuongnien_{year}.pdf"


def exchange_of(ticker: str) -> str:
    with open(ROOT / "reference/companies.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["ticker"] == ticker:
                return row["exchange"] or "HOSE"
    return "HOSE"


def fetch(ticker: str, year: int) -> str:
    target = ROOT / f"data/reports/{ticker}/{ticker}_{year}.pdf"
    if target.exists():
        return "đã có"
    url = URL.format(exchange=exchange_of(ticker), year=year, ticker=ticker)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
    except urllib.error.HTTPError as exc:
        return f"không có ({exc.code})"
    if not data.startswith(b"%PDF"):
        return "không phải PDF, bỏ qua"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return f"tải {len(data) / 1e6:.1f} MB"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--years", default="2021-2025")
    args = ap.parse_args()
    first, _, last = args.years.partition("-")
    for ticker in (t.upper() for t in args.tickers):
        for year in range(int(first), int(last or first) + 1):
            print(f"{ticker} {year}: {fetch(ticker, year)}")
            sys.stdout.flush()
