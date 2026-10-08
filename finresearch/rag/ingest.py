"""Turn an annual report PDF into page-anchored chunks.

Annual reports are designed documents: slide-like pages, several columns, a
navigation sidebar repeated on every page and text stored out of reading order.
This module puts each page back into reading order, drops the repeated chrome,
tracks the current heading, and cuts chunks that never cross a page so every
chunk can be cited by one page number.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pymupdf

CHUNK_WORDS = 300      # target size; Vietnamese syllables, roughly one token each
MAX_WORDS = 380
OVERLAP_WORDS = 50
MIN_PAGE_WORDS = 40    # covers, dividers and photo pages
REPEAT_PAGES = 5       # a line seen at the same spot on this many pages is navigation, a header or a footer
NUMERIC_SHARE = 0.35   # above this share of digits a page is a financial statement, not prose
TABLE_SHARE = 0.10     # between the two, the page mixes prose with a table or chart worth reading row by row
ROW_TOLERANCE = 0.012  # lines this close vertically (share of page height) sit on the same table row


@dataclass
class Line:
    text: str
    size: float
    x: float
    y: float


@dataclass
class Chunk:
    id: str
    ticker: str
    year: int
    page: int       # 1-based page of the PDF file, as a PDF viewer shows it
    section: str
    kind: str       # "text", "table" (same page read row by row) or "numeric" (statement page, not searched)
    text: str

    @property
    def n_words(self) -> int:
        return len(self.text.split())

    def embed_text(self) -> str:
        """What the encoder sees: the chunk plus where it comes from."""
        head = f"{self.ticker} | Báo cáo thường niên {self.year}"
        return f"{head} | {self.section}\n{self.text}" if self.section else f"{head}\n{self.text}"


def _norm(text: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text).strip().lower())


def _chrome_key(line: Line, width: float, height: float) -> tuple:
    """Text plus rough position. Position matters: "Doanh thu" recurs as a table label
    anywhere on a page, while a menu entry or a footer always sits in the same place."""
    return _norm(line.text), round(50 * line.x / width), round(50 * line.y / height)


def _page_lines(page) -> list[Line]:
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = re.sub(r"\s+", " ", "".join(s["text"] for s in line["spans"])).strip()
            size = max(spans, key=lambda s: len(s["text"]))["size"]
            lines.append(Line(text=text, size=size, x=line["bbox"][0], y=line["bbox"][1]))
    return lines


def _reading_order(lines: list[Line], page_width: float) -> list[Line]:
    """Group lines into columns by their left edge, then read each column top to bottom."""
    if not lines:
        return []
    tolerance = 0.06 * page_width
    columns: list[list[Line]] = []
    for line in sorted(lines, key=lambda l: l.x):
        if columns and line.x - columns[-1][0].x <= tolerance:
            columns[-1].append(line)
        else:
            columns.append([line])
    return [l for col in columns for l in sorted(col, key=lambda l: (round(l.y), l.x))]


def _rows(lines: list[Line], page_height: float) -> list[str]:
    """Read the page across instead of down, so each table row keeps its label and its figures together."""
    rows: list[list[Line]] = []
    for line in sorted(lines, key=lambda l: (l.y, l.x)):
        if rows and abs(line.y - rows[-1][0].y) <= ROW_TOLERANCE * page_height:
            rows[-1].append(line)
        else:
            rows.append([line])
    return [" | ".join(l.text for l in sorted(row, key=lambda l: l.x)) for row in rows]


def _pack_rows(rows: list[str]) -> list[str]:
    """Join rows into pieces of at most MAX_WORDS without cutting a row."""
    parts, current, size = [], [], 0
    for row in rows:
        n = len(row.split())
        if current and size + n > MAX_WORDS:
            parts.append("\n".join(current))
            current, size = [], 0
        current.append(row)
        size += n
    return parts + (["\n".join(current)] if current else [])


def _is_heading(line: Line, body_size: float) -> bool:
    words = line.text.split()
    if not (1.3 * body_size <= line.size <= 3.0 * body_size) or not (2 <= len(words) <= 18):
        return False
    letters = sum(ch.isalpha() for ch in line.text)
    return letters >= 0.6 * len(line.text.replace(" ", "")) and not line.text.endswith((".", ",", ";"))


def _split_words(words: list[str]) -> list[list[str]]:
    """Cut a page's words into overlapping pieces of about CHUNK_WORDS."""
    if len(words) <= MAX_WORDS:
        return [words]
    n_parts = -(-len(words) // CHUNK_WORDS)
    step = -(-len(words) // n_parts)
    return [words[max(0, i - OVERLAP_WORDS): i + step] for i in range(0, len(words), step)]


def parse_pdf(path: str | Path, ticker: str, year: int) -> tuple[list[Chunk], dict]:
    """Return the chunks of one report and a few facts about the file."""
    path = Path(path)
    doc = pymupdf.open(path)
    pages = [_page_lines(p) for p in doc]

    dims = [(p.rect.width, p.rect.height) for p in doc]
    sizes = Counter()
    seen_on = defaultdict(set)
    for i, lines in enumerate(pages):
        for l in lines:
            sizes[round(l.size)] += len(l.text)
            seen_on[_chrome_key(l, *dims[i])].add(i)
    body_size = sizes.most_common(1)[0][0] if sizes else 10
    # only lines with words can be chrome: a bare number repeats by coincidence, and
    # keeping table cells is what lets the digit share below spot statement pages
    chrome = {k for k, where in seen_on.items() if len(where) >= REPEAT_PAGES and any(ch.isalpha() for ch in k[0])}

    chunks: list[Chunk] = []
    section = ""
    for i, lines in enumerate(pages):
        kept = [l for l in lines if _chrome_key(l, *dims[i]) not in chrome and not re.fullmatch(r"[\d\W]{1,4}", l.text)]
        ordered = _reading_order(kept, doc[i].rect.width)
        headings = [l.text for l in ordered if _is_heading(l, body_size)]
        page_section = headings[0] if headings else section
        text = " ".join(l.text for l in ordered)
        words = text.split()
        if headings:
            section = headings[-1]
        if len(words) < MIN_PAGE_WORDS:
            continue
        digits = sum(ch.isdigit() for ch in text)
        letters = sum(ch.isalpha() for ch in text)
        share = digits / max(digits + letters, 1)
        kind = "numeric" if share > NUMERIC_SHARE else "text"
        base = dict(ticker=ticker, year=year, page=i + 1, section=page_section[:200])
        for n, part in enumerate(_split_words(words), 1):
            chunks.append(Chunk(id=f"{ticker}-AR{year}-p{i + 1:03d}-c{n}", kind=kind, text=" ".join(part), **base))
        if TABLE_SHARE < share <= NUMERIC_SHARE:
            for n, part in enumerate(_pack_rows(_rows(kept, doc[i].rect.height)), 1):
                chunks.append(Chunk(id=f"{ticker}-AR{year}-p{i + 1:03d}-t{n}", kind="table", text=part, **base))
    info = {"file": path.name, "n_pages": len(doc), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "body_size": body_size, "chrome_lines": len(chrome)}
    return chunks, info
