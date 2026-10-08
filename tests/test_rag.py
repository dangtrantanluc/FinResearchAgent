import pymupdf
import pytest

from finresearch.rag.ingest import Line, _reading_order, parse_pdf
from finresearch.rag.search import _fuse



def _body(page: int, side: str) -> str:
    """Prose that differs on every page; identical lines would be mistaken for page chrome."""
    tag = side + "abcdefgh"[page]  # letters only: the chunker treats digits as interchangeable
    return " ".join(f"{tag}word{chr(97 + j % 26)}{chr(97 + j // 26)}" for j in range(110))


def _make_pdf(path, n_pages=6):
    doc = pymupdf.open()
    for i in range(n_pages):
        page = doc.new_page(width=1000, height=700)
        page.insert_text((40, 30), "ANNUAL REPORT 2024", fontsize=10)          # header on every page
        page.insert_text((40, 60), f"{i + 1}", fontsize=10)                     # page number
        if i == 4:                                                              # a statement page
            for row in range(40):
                page.insert_text((40, 100 + 14 * row), f"Line item {row}", fontsize=10)
                page.insert_text((400, 100 + 14 * row), "1.234.567.890.123   9.876.543.210.987", fontsize=10)
            continue
        page.insert_text((40, 100), f"SEGMENT REVIEW {'ABCDEF'[i]}", fontsize=20)
        page.insert_textbox(pymupdf.Rect(40, 120, 480, 680), f"Left column of page {i + 1}. " + _body(i, "l"), fontsize=10)
        page.insert_textbox(pymupdf.Rect(520, 120, 960, 680), f"Right column of page {i + 1}. " + _body(i, "r"), fontsize=10)
    doc.save(path)


def test_reading_order_reads_a_column_before_moving_right():
    lines = [Line("right top", 10, 520, 100), Line("left bottom", 10, 40, 300), Line("left top", 10, 42, 100),
             Line("right bottom", 10, 521, 300)]
    assert [l.text for l in _reading_order(lines, page_width=1000)] == ["left top", "left bottom", "right top", "right bottom"]


def test_parse_pdf_strips_chrome_and_keeps_page_numbers(tmp_path):
    _make_pdf(tmp_path / "X_2024.pdf")
    chunks, info = parse_pdf(tmp_path / "X_2024.pdf", "X", 2024)
    assert info["n_pages"] == 6
    assert {c.page for c in chunks} == {1, 2, 3, 4, 5, 6}
    assert all("ANNUAL REPORT" not in c.text for c in chunks)                   # repeated header removed
    first = next(c for c in chunks if c.page == 1)
    assert first.id == "X-AR2024-p001-c1" and first.section == "SEGMENT REVIEW A"
    assert first.text.index("Left column of page 1") < first.text.index("Right column of page 1")
    assert {c.kind for c in chunks if c.page == 5} == {"numeric"}               # statement page is flagged
    assert all(c.kind == "text" for c in chunks if c.page != 5)


def test_long_pages_are_split_with_overlap(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=1000, height=1400)
    page.insert_textbox(pymupdf.Rect(40, 40, 960, 1380), " ".join(f"w{i}" for i in range(900)), fontsize=8)
    doc.save(tmp_path / "Y_2024.pdf")
    chunks, _ = parse_pdf(tmp_path / "Y_2024.pdf", "Y", 2024)
    assert len(chunks) == 3 and all(c.n_words <= 380 for c in chunks)
    assert set(chunks[0].text.split()) & set(chunks[1].text.split())             # neighbours share words
    assert [c.id[-2:] for c in chunks] == ["c1", "c2", "c3"]


def test_rrf_rewards_agreement_between_rankings():
    scores = _fuse([["a", "b", "c"], ["b", "d", "a"]])
    assert max(scores, key=scores.get) == "b" and scores["a"] > scores["c"] and scores["a"] > scores["d"]


def _has_chunks() -> bool:
    try:
        from finresearch.db import query
        return int(query("SELECT count(*) AS n FROM chunks WHERE ticker = 'FPT'").n.iloc[0]) > 0
    except Exception:
        return False


@pytest.mark.skipif(not _has_chunks(), reason="reports not ingested (scripts/ingest_reports.py)")
def test_lexical_search_finds_a_named_entity_on_its_page():
    from finresearch.rag.search import search

    hits = search("đầu tư chiến lược vào Base.vn", ticker="FPT", years=[2021], mode="lexical")
    assert hits and all(h.year == 2021 and h.ticker == "FPT" for h in hits)
    assert {h.page for h in hits} & {4, 8, 14, 26, 31}
    assert hits[0].citation() == f"BCTN FPT 2021, tr. {hits[0].page}"
