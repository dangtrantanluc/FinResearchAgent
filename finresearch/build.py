"""Run the extractor over the whole dataset and pick one report per ticker-year."""
from __future__ import annotations

import re
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .extract import extract_file, fold

KINDS = ("BS", "IS", "CF")
CONSOLIDATED_NAME = re.compile(r"hop ?nhat|hopnhat|consolidat|(?<![a-z])hn(?![a-z])")
PARENT_NAME = re.compile(r"cong ?ty ?me|congtyme|rieng|separate|tong ?hop|(?<![a-z])me(?![a-z])")


def _rel_err(a, b) -> float:
    """Relative gap between two amounts that should be equal; nan if either is missing."""
    if a is None or b is None:
        return np.nan
    scale = max(abs(a), abs(b))
    return 0.0 if scale == 0 else abs(a - b) / scale


def _sum(st, codes, which):
    vals = [st.get(c, which) for c in codes]
    if vals[0] is None:
        return None
    return sum(v or 0.0 for v in vals)


def identity_errors(report) -> dict[str, float]:
    """Accounting identities, worst of the two year columns. Small means consistent."""
    out = {}
    bs, is_, cf = (report.statements.get(k) for k in KINDS)
    worst = lambda errs: np.nan if all(np.isnan(e) for e in errs) else np.nanmax(errs)
    if bs:
        out["err_assets"] = worst([_rel_err(bs.get("270", w), _sum(bs, ["100", "200"], w)) for w in (0, 1)])
        out["err_funding"] = worst([_rel_err(bs.get("440", w), _sum(bs, ["300", "400"], w)) for w in (0, 1)])

        def balance(w):  # a total row is sometimes lost in OCR, so accept the sum of its parts
            assets = [bs.get("270", w), _sum(bs, ["100", "200"], w)]
            funding = [bs.get("440", w), _sum(bs, ["300", "400"], w)]
            errs = [_rel_err(a, f) for a in assets for f in funding]
            return np.nan if all(np.isnan(e) for e in errs) else np.nanmin(errs)

        out["err_balance"] = worst([balance(w) for w in (0, 1)])
    if is_:
        # cost of sales is printed as a positive amount by some auditors and in brackets by others
        gross = lambda w: None if is_.get("10", w) is None else is_.get("10", w) - abs(is_.get("11", w) or 0.0)
        out["err_gross"] = worst([_rel_err(is_.get("20", w), gross(w)) for w in (0, 1)])
    if cf:
        out["err_cash"] = worst([_rel_err(cf.get("50", w), _sum(cf, ["20", "30", "40"], w)) for w in (0, 1)])
    return out


def parse_path(path: Path, raw_dir: Path) -> dict:
    ticker, year, stem = path.relative_to(raw_dir).parts[:3]
    name = fold(stem.replace("_", " ").replace("-", " ").replace(".", " "))
    if CONSOLIDATED_NAME.search(name):
        scope = "consolidated"
    elif PARENT_NAME.search(name):
        scope = "parent"
    else:
        scope = "unknown"
    return {"ticker": ticker, "year": int(year), "file": str(path.relative_to(raw_dir)), "name_scope": scope}


def process_file(args) -> tuple[dict, list[tuple]]:
    path, raw_dir = args
    meta = parse_path(path, raw_dir)
    try:
        report = extract_file(path)
    except Exception as exc:  # one broken file must not stop an 18k-file run
        meta["error"] = f"{type(exc).__name__}: {exc}"[:200]
        return meta, []
    meta.update(
        n_tables=report.n_tables,
        n_coded_tables=report.n_coded_tables,
        consolidated_title=report.consolidated_title,
        english=report.english,
        **identity_errors(report),
    )
    rows = []
    for kind in KINDS:
        st = report.statements.get(kind)
        meta[f"n_{kind}"] = len(st.lines) if st else 0
        meta[f"label_{kind}"] = report.label_match(kind)
        if st:
            meta[f"unit_{kind}"] = st.unit
            for code, (label, cur, prev) in st.lines.items():
                rows.append((meta["file"], kind, code, label[:80], cur, prev))
    return meta, rows


def extract_all(raw_dir: str | Path, workers: int = 4, limit: int | None = None):
    """Return (reports, facts): one row per file, and one row per extracted line."""
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("*/*/*/*_extracted.txt"))[:limit]
    jobs = [(p, raw_dir) for p in files]
    if workers > 1:
        with Pool(workers) as pool:
            results = pool.map(process_file, jobs, chunksize=32)
    else:
        results = [process_file(j) for j in jobs]
    reports = pd.DataFrame([m for m, _ in results])
    facts = pd.DataFrame(
        [r for _, rows in results for r in rows],
        columns=["file", "statement", "code", "label", "cur", "prev"],
    )
    return reports, facts


TOL = 0.005  # identities must hold within 0.5%


def grade_reports(reports: pd.DataFrame) -> pd.DataFrame:
    """Add quality flags and the final scope of each report."""
    r = reports.copy()
    for col in ("err_assets", "err_funding", "err_balance", "err_gross", "err_cash"):
        if col not in r:
            r[col] = np.nan
    r["bs_ok"] = (r.err_balance <= TOL) & (r.label_BS >= 0.6)
    r["is_ok"] = (r.err_gross <= TOL) & (r.label_IS >= 0.8)
    r["cf_ok"] = (r.err_cash <= TOL) & (r.label_CF >= 0.6)
    # Circular 200 layout: excludes banks, brokers and insurers.
    r["corporate"] = (r.label_IS >= 0.8) & (r.label_BS >= 0.6)

    scope = r.name_scope.where(r.name_scope != "unknown", np.where(r.consolidated_title, "consolidated", "unknown"))
    # A ticker that ever files consolidated statements is a group, so its
    # other statements are the parent company's own accounts.
    is_group = scope.eq("consolidated").groupby(r.ticker).transform("any")
    r["scope"] = np.where(scope == "consolidated", "consolidated", np.where(is_group, "parent", "single"))
    r["quality"] = r.bs_ok.astype(int) + r.is_ok.astype(int) + r.cf_ok.astype(int)
    return r


def select_reports(graded: pd.DataFrame) -> pd.DataFrame:
    """One report per ticker-year: consolidated first, then best quality."""
    rank = graded.scope.map({"consolidated": 0, "single": 0, "parent": 1})
    g = graded.assign(_rank=rank, _lines=graded.n_BS + graded.n_IS + graded.n_CF)
    g = g[g.corporate].sort_values(
        ["ticker", "year", "_rank", "quality", "english", "_lines"],
        ascending=[True, True, True, False, True, False],
    )
    return g.drop_duplicates(["ticker", "year"]).drop(columns=["_rank", "_lines"])


DATASET_URL = "https://huggingface.co/datasets/vduydong/ocr_annual_financials"


def download_raw(raw_dir: str | Path) -> Path:
    """Shallow git clone: one request instead of 18k file downloads."""
    raw_dir = Path(raw_dir)
    if not any(raw_dir.glob("*/*/*/*_extracted.txt")):
        raw_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", DATASET_URL, str(raw_dir)], check=True)
    return raw_dir


def build_dataset(raw_dir: str | Path, out_dir: str | Path, companies_csv: str | Path | None = None, workers: int = 4):
    """Extract, grade, select and assemble. Writes parquet files and returns the panel."""
    from .panel import build_panel, build_wide

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports, facts = extract_all(raw_dir, workers=workers)
    graded = grade_reports(reports)
    selected = select_reports(graded)
    companies = pd.read_csv(companies_csv) if companies_csv and Path(companies_csv).exists() else None
    wide = build_wide(facts, selected)
    panel = build_panel(wide, companies)
    graded.to_parquet(out_dir / "reports.parquet", index=False)
    facts[facts.file.isin(selected.file)].to_parquet(out_dir / "facts.parquet", index=False)
    wide.reset_index().to_parquet(out_dir / "fundamentals.parquet", index=False)
    panel.to_parquet(out_dir / "panel.parquet", index=False)
    return graded, selected, panel
