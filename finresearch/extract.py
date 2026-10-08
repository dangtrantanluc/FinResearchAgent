"""Extract coded financial statement lines from OCR'd Vietnamese audited reports.

Input is one `*_extracted.txt` file from the `vduydong/ocr_annual_financials`
dataset. The statements appear as HTML tables whose rows carry the line code
("Mã số") defined by Circular 200/2014/TT-BTC, plus a current-year and a
prior-year amount. The code column, the label layout and the number format all
vary between auditors, so nothing here assumes a fixed column position.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

TABLE_RE = re.compile(r"<table.*?</table>", re.S | re.I)
TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
TD_RE = re.compile(r"<t[dh]([^>]*)>(.*?)</t[dh]>", re.S | re.I)
SPAN_RE = re.compile(r"(row|col)span\s*=\s*[\"']?(\d+)", re.I)
TAG_RE = re.compile(r"<[^>]+>")
CODE_RE = re.compile(r"^\d{1,3}[a-zA-Z]?$")
GROUPED_RE = re.compile(r"^\d{1,3}([.,]\d{3})+$")
DECIMAL_RE = re.compile(r"^\d+[.,]\d{1,2}$")
YEAR_RE = re.compile(r"(?<!\d)(20[0-3]\d)(?!\d)")

CONTEXT_LINES = 8


def fold(s: str) -> str:
    """Lowercase and strip diacritics so OCR accent errors do not matter."""
    s = s.replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", s).lower().strip()


def parse_number(cell: str) -> float | None:
    """Parse an amount cell. Returns None when the cell is not an amount.

    Handles `1.234.567`, `1,234,567`, `(1.234)`, `-1.234` and a lone dash (nil).
    """
    s = cell.strip().replace("\xa0", "").replace(" ", "")
    if not s:
        return None
    if s in {"-", "–", "—", "_", "0"}:
        return 0.0
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1]
    elif s.startswith("(") or s.endswith(")"):
        neg, s = True, s.strip("()")
    if s[:1] in "-–—":
        neg, s = True, s[1:]
    if GROUPED_RE.match(s):
        v = float(re.sub(r"[.,]", "", s))
    elif s.isdigit():
        v = float(s)
    elif DECIMAL_RE.match(s):
        v = float(s.replace(",", "."))
    else:
        return None
    return -v if neg else v


@dataclass
class Table:
    rows: list[list[str]]
    context: str  # folded text of the lines just above the table
    line: int


def _grid(table_html: str) -> list[list[str]]:
    """Expand rowspan/colspan so every row has the same column positions."""
    grid: list[list[str]] = []
    carry: dict[int, tuple[int, str]] = {}  # col -> (rows left, text)
    for tr in TR_RE.findall(table_html):
        row: list[str] = []
        col = 0
        cells = TD_RE.findall(tr)
        for attrs, inner in cells:
            while col in carry:
                left, _ = carry[col]
                row.append("")
                carry[col] = (left - 1, "")
                if carry[col][0] <= 0:
                    del carry[col]
                col += 1
            spans = {k.lower(): int(v) for k, v in SPAN_RE.findall(attrs)}
            text = re.sub(r"\s+", " ", TAG_RE.sub(" ", inner)).strip()
            cs, rs = max(spans.get("col", 1), 1), max(spans.get("row", 1), 1)
            for k in range(cs):
                row.append(text if k == 0 else "")
                if rs > 1:
                    carry[col] = (rs - 1, "")
                col += 1
        while col in carry:
            left, _ = carry[col]
            row.append("")
            carry[col] = (left - 1, "")
            if carry[col][0] <= 0:
                del carry[col]
            col += 1
        if row:
            grid.append(row)
    return grid


def iter_tables(text: str):
    for m in TABLE_RE.finditer(text):
        rows = _grid(m.group(0))
        if not rows:
            continue
        before = TABLE_RE.sub(" ", text[max(0, m.start() - 3000):m.start()])
        ctx = [l.strip() for l in before.split("\n") if l.strip()]
        context = fold(" // ".join(ctx[-CONTEXT_LINES:]))
        yield Table(rows=rows, context=context, line=text.count("\n", 0, m.start()))


CODE_HEADER_RE = re.compile(r"\bma ?so\b|^ms$|^m\.?s\.?$|\bcodes?\b|^ma$")


def find_code_col(rows: list[list[str]]) -> int | None:
    """Column that holds the line code, or None if the table has no codes."""
    width = max(len(r) for r in rows)
    counts = Counter()
    for r in rows:
        for j, c in enumerate(r[: min(width, 5)]):
            if CODE_RE.match(c.strip()):
                counts[j] += 1
    if not counts or max(counts.values()) < 3:
        return None
    for r in rows[:3]:  # a header naming the column wins over content counts
        for j, c in enumerate(r[: min(width, 5)]):
            if CODE_HEADER_RE.search(fold(c)) and counts.get(j, 0) >= 3:
                return j
    best = max(counts.values())
    # The note column also holds small integers; it is sparser and to the right.
    return min(j for j, n in counts.items() if n >= 0.8 * best)


CUR_HINT = re.compile(r"cuoi nam|cuoi ky|nam nay|ky nay|so cuoi|closing|current year|ending")
PREV_HINT = re.compile(r"dau nam|dau ky|nam truoc|ky truoc|so dau|opening|previous year|prior year|beginning")


def find_value_cols(rows: list[list[str]], code_col: int) -> tuple[int, int, bool] | None:
    """Return (cur_col, prev_col, order_from_header) for a coded table."""
    width = max(len(r) for r in rows)
    coded = [r for r in rows if len(r) > code_col and CODE_RE.match(r[code_col].strip())]
    numeric = Counter()
    for r in coded:
        for j in range(code_col + 1, min(len(r), width)):
            v = parse_number(r[j])
            # amounts only: note references such as "5" or "V.1" must not count
            if v is not None and (abs(v) >= 1000 or r[j].strip() in {"-", "–", "—"}):
                numeric[j] += 1
    cols = sorted(j for j, n in numeric.items() if n >= max(2, 0.3 * len(coded)))
    if len(cols) < 2:
        return None
    a, b = cols[-2], cols[-1]

    header = [r for r in rows[:4] if not (len(r) > code_col and CODE_RE.match(r[code_col].strip()))]
    ha = fold(" ".join(r[a] for r in header if len(r) > a))
    hb = fold(" ".join(r[b] for r in header if len(r) > b))
    ya = [int(y) for y in YEAR_RE.findall(ha)]
    yb = [int(y) for y in YEAR_RE.findall(hb)]
    if ya and yb and max(ya) != max(yb):
        return (a, b, True) if max(ya) > max(yb) else (b, a, True)
    if PREV_HINT.search(ha) and CUR_HINT.search(hb):
        return b, a, True
    if CUR_HINT.search(ha) and PREV_HINT.search(hb):
        return a, b, True
    return a, b, False


BS_TITLE = re.compile(r"can doi ke toan|balance sheet|financial position|tinh hinh tai chinh|b ?01")
IS_TITLE = re.compile(r"ket qua (hoat dong )?kinh doanh|income statement|profit (or|and) loss|b ?02")
CF_TITLE = re.compile(r"luu chuyen tien|cash flow|b ?03")
IS_WORDS = re.compile(
    r"doanh thu ban hang|doanh thu thuan|gia von|loi nhuan gop|chi phi ban hang|chi phi quan ly|"
    r"thu nhap khac|lai co ban|lai suy giam|chi phi thue|net revenue|net sales|cost of|gross profit|"
    r"selling expense|administrati|earnings per share|other income"
)
CF_WORDS = re.compile(
    r"luu chuyen|khau hao|tien chi|tien thu|tang,? ?giam|\(tang\)|tien va (cac khoan )?tuong duong|"
    r"cash flow|net cash|depreciation|payments? |proceeds|increase|decrease|cash and cash equivalents at"
)


LETTERS_RE = re.compile(r"[^\W\d_]")


def row_label(row: list[str], code_col: int) -> str:
    """Text of a row: every cell that reads as words, whichever side of the code it is on."""
    parts = [c for j, c in enumerate(row) if j != code_col and len(LETTERS_RE.findall(c)) >= 3]
    return " ".join(parts).strip()


def classify(rows: list[list[str]], code_col: int, context: str) -> str | None:
    """Decide whether a coded table is a balance sheet, income statement or cash flow."""
    codes, labels = [], []
    for r in rows:
        if len(r) > code_col and CODE_RE.match(r[code_col].strip()):
            codes.append(r[code_col].strip())
            labels.append(fold(row_label(r, code_col)))
    if len(codes) < 3:
        return None
    three_digit = sum(1 for c in codes if len(re.sub(r"\D", "", c)) == 3)
    if three_digit >= 0.6 * len(codes):
        return "BS"
    if three_digit > 0.3 * len(codes):
        return None  # mixed codes: not one of the three statements
    is_score = sum(1 for l in labels if IS_WORDS.search(l))
    cf_score = sum(1 for l in labels if CF_WORDS.search(l))
    tail = context[-260:]
    if IS_TITLE.search(tail) and not CF_TITLE.search(tail):
        is_score += 2
    if CF_TITLE.search(tail) and not IS_TITLE.search(tail):
        cf_score += 2
    if is_score == cf_score == 0:
        return None
    return "IS" if is_score > cf_score else "CF"


UNIT_PATTERNS = [
    (re.compile(r"ty (dong|vnd)|billion"), 1e9),
    (re.compile(r"trieu (dong|vnd)|million|vnd ?million|tr\.? ?d\b"), 1e6),
    (re.compile(r"(nghin|ngan|1\.000|1000) (dong|vnd)|thousand|'000"), 1e3),
]


def detect_unit(text: str) -> float:
    for pat, mult in UNIT_PATTERNS:
        if pat.search(text):
            return mult
    return 1.0


# Total rows are sometimes printed without a code; recover them from the label.
LABEL_CODES = {
    "BS": [
        (re.compile(r"^tong (cong )?tai san\b|^total assets"), "270"),
        (re.compile(r"^tong (cong )?nguon von\b|^total (resources|liabilities and)"), "440"),
    ],
}

# Labels a key code must carry in a Circular 200 statement. Banks, brokers and
# insurers reuse the same numbers for different lines and fail this check.
KEY_LABELS = {
    "IS": {
        "10": r"doanh thu thuan|net revenue|net sales|net turnover",
        "11": r"gia von|cost of",
        "20": r"loi nhuan gop|lo gop|gross",
        "50": r"truoc thue|before tax",
        "60": r"sau thue|after tax|net profit|profit for the",
    },
    "BS": {
        "100": r"ngan han|current assets",
        "200": r"dai han|non-?current|long-?term",
        "270": r"tong (cong )?tai san|total assets",
        "300": r"no phai tra|liabilities",
        "400": r"von chu so huu|equity|nguon von",
        "440": r"tong (cong )?nguon von|total",
    },
    "CF": {
        "20": r"hoat dong (san xuat,? ?)?kinh doanh|operating",
        "30": r"dau tu|investing",
        "40": r"tai chinh|financing",
    },
}


def norm_code(c: str) -> str:
    c = c.strip().lower()
    digits = re.sub(r"\D", "", c)
    suffix = c[len(digits):]
    if len(digits) == 1:
        digits = "0" + digits
    return digits + suffix


@dataclass
class Statement:
    lines: dict[str, tuple[str, float | None, float | None]] = field(default_factory=dict)
    unit: float = 1.0
    order_from_header: bool = False
    n_tables: int = 0
    title: str = ""

    def get(self, code: str, which: int) -> float | None:
        v = self.lines.get(code)
        return None if v is None else v[1 + which]


@dataclass
class Report:
    statements: dict[str, Statement]
    n_tables: int
    n_coded_tables: int
    consolidated_title: bool
    english: bool

    def label_match(self, kind: str) -> float:
        """Share of key codes whose label matches the Circular 200 wording."""
        st = self.statements.get(kind)
        if st is None:
            return 0.0
        keys = KEY_LABELS[kind]
        hits = sum(1 for c, pat in keys.items() if c in st.lines and re.search(pat, fold(st.lines[c][0])))
        return hits / len(keys)


def extract_report(text: str) -> Report:
    statements: dict[str, Statement] = {}
    n_tables = n_coded = 0
    for table in iter_tables(text):
        n_tables += 1
        code_col = find_code_col(table.rows)
        if code_col is None:
            continue
        kind = classify(table.rows, code_col, table.context)
        value_cols = find_value_cols(table.rows, code_col)
        if kind is None or value_cols is None:
            continue
        n_coded += 1
        cur_col, prev_col, from_header = value_cols
        st = statements.setdefault(kind, Statement())
        if st.n_tables == 0:
            header_text = fold(" ".join(" ".join(r) for r in table.rows[:3]))
            st.unit = detect_unit(table.context[-200:] + " " + header_text)
            st.order_from_header = from_header
            st.title = table.context[-200:]
        st.n_tables += 1
        for r in table.rows:
            if len(r) <= max(code_col, cur_col, prev_col):
                continue
            raw = r[code_col].strip()
            label = row_label(r, code_col)
            if CODE_RE.match(raw):
                code = norm_code(raw)
            else:
                code = next((c for pat, c in LABEL_CODES.get(kind, []) if pat.search(fold(label))), None)
                if code is None:
                    continue
            if code in st.lines:  # keep the first occurrence; later ones are repeats
                continue
            st.lines[code] = (label, parse_number(r[cur_col]), parse_number(r[prev_col]))
    head = fold(text[:4000])
    titles = " ".join(s.title for s in statements.values())
    return Report(
        statements=statements,
        n_tables=n_tables,
        n_coded_tables=n_coded,
        consolidated_title=bool(re.search(r"hop nhat|consolidated", titles + " " + head)),
        english=bool(re.search(r"balance sheet|financial position|income statement", titles)),
    )


def extract_file(path: str | Path) -> Report:
    return extract_report(Path(path).read_text(encoding="utf-8", errors="replace"))
