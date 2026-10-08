"""Parse, embed and store every report under data/reports/<TICKER>/<TICKER>_<YEAR>.pdf.

    python scripts/ingest_reports.py [TICKER ...]
"""
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finresearch.db import store_document  # noqa: E402
from finresearch.rag.ingest import parse_pdf  # noqa: E402
from finresearch.rag.models import embed  # noqa: E402

if __name__ == "__main__":
    only = {t.upper() for t in sys.argv[1:]}
    for path in sorted((ROOT / "data/reports").glob("*/*.pdf")):
        m = re.fullmatch(r"([A-Z0-9]+)_(\d{4})\.pdf", path.name)
        if not m or (only and m.group(1) not in only):
            continue
        ticker, year = m.group(1), int(m.group(2))
        start = time.time()
        chunks, info = parse_pdf(path, ticker, year)
        vectors = embed([c.embed_text() for c in chunks])
        store_document(ticker, year, chunks, vectors, info)
        n = {k: sum(c.kind == k for c in chunks) for k in ("text", "table", "numeric")}
        print(f"{path.name}: {info['n_pages']} trang → {n['text']} chunk văn bản, {n['table']} chunk bảng, "
              f"{n['numeric']} chunk báo cáo tài chính (không tìm kiếm) trong {time.time() - start:.0f}s")
