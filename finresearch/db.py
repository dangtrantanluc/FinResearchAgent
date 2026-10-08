"""Postgres access and the loader for everything the build step produces."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import psycopg

DEFAULT_URL = "postgresql://finresearch:finresearch@127.0.0.1:5433/finresearch"

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
DROP TABLE IF EXISTS companies, reports, fundamentals, statement_lines, metrics, ratios, forecasts, scope_changes;
CREATE TABLE companies (
    ticker text PRIMARY KEY, name text, exchange text, com_type text, icb1 text, icb2 text, icb3 text
);
-- the one audited report chosen for each ticker-year
CREATE TABLE reports (
    ticker text, year int, file text, scope text, bs_ok bool, is_ok bool, cf_ok bool, usable bool,
    PRIMARY KEY (ticker, year)
);
-- named line items in VND, sign-normalised (see finresearch.panel.FIELDS)
CREATE TABLE fundamentals (
    ticker text, year int, field text, value double precision, value_prev double precision,
    PRIMARY KEY (ticker, year, field)
);
-- every coded line of the three statements, as printed
CREATE TABLE statement_lines (
    ticker text, year int, statement text, code text, label text, value double precision, value_prev double precision,
    PRIMARY KEY (ticker, year, statement, code)
);
-- model inputs: the features exactly as the growth model was trained on them, clipped to sane ranges
CREATE TABLE metrics (
    ticker text, year int, name text, value double precision,
    PRIMARY KEY (ticker, year, name)
);
-- the same ratios as computed, unclipped: what a reader is shown. A firm whose debt grew tenfold
-- must not be reported at the model's cap of +500%.
CREATE TABLE ratios (
    ticker text, year int, name text, value double precision,
    PRIMARY KEY (ticker, year, name)
);
CREATE TABLE forecasts (
    ticker text, target_year int, pred_rel double precision, vol_group text, radius double precision,
    growth_lo double precision, growth_mid double precision, growth_hi double precision,
    PRIMARY KEY (ticker, target_year)
);
-- changes in what a group consolidates; a growth model cannot see these coming
CREATE TABLE scope_changes (
    ticker text, effective_year int, kind text, counterpart text, note text, note_en text, source text
);
CREATE INDEX ratios_name_year ON ratios (name, year);
"""

# Report text lives beside the numbers but is loaded separately (scripts/ingest_reports.py),
# so load_all() never touches these tables. No ANN index: a few thousand chunks are
# scanned exactly in milliseconds, and exact search has no recall loss under filters.
RAG_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS documents (
    id serial PRIMARY KEY, ticker text, year int, doc_type text, file text, sha256 text, n_pages int,
    UNIQUE (ticker, year, doc_type)
);
CREATE TABLE IF NOT EXISTS chunks (
    id text PRIMARY KEY, document_id int REFERENCES documents(id) ON DELETE CASCADE,
    ticker text, year int, page int, section text, kind text, text text, embedding vector(1024)
);
CREATE INDEX IF NOT EXISTS chunks_ticker_year ON chunks (ticker, year);
"""


def connect() -> psycopg.Connection:
    return psycopg.connect(os.environ.get("DATABASE_URL", DEFAULT_URL))


def query(sql: str, params: tuple | dict | None = None, conn: psycopg.Connection | None = None) -> pd.DataFrame:
    own = conn is None
    conn = conn or connect()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    finally:
        if own:
            conn.close()


def _copy(cur, table: str, df: pd.DataFrame) -> int:
    df = df.astype(object).where(df.notna(), None)
    with cur.copy(f"COPY {table} ({', '.join(df.columns)}) FROM STDIN") as cp:
        for row in df.itertuples(index=False, name=None):
            cp.write_row(row)
    return len(df)


def load_all(root: str | Path) -> dict[str, int]:
    """Rebuild every table from data/processed, reference/ and artifacts/. Returns row counts."""
    from .build import select_reports
    from .panel import FIELDS, add_features, feature_columns

    root = Path(root)
    processed = root / "data/processed"
    graded = pd.read_parquet(processed / "reports.parquet")
    facts = pd.read_parquet(processed / "facts.parquet")
    wide = pd.read_parquet(processed / "fundamentals.parquet")
    panel = pd.read_parquet(processed / "panel.parquet")
    selected = select_reports(graded)

    companies = pd.read_csv(root / "reference/companies.csv")

    reports = selected[["ticker", "year", "file", "scope", "bs_ok", "is_ok", "cf_ok"]].merge(
        panel[["ticker", "year", "usable"]], on=["ticker", "year"], how="left"
    )

    fields = [f for f in FIELDS if f in wide.columns]
    cur_ = wide.melt(["ticker", "year"], fields, "field", "value")
    prev = wide.melt(["ticker", "year"], [f + "_prev" for f in fields], "field", "value_prev")
    prev["field"] = prev.field.str.removesuffix("_prev")
    fundamentals = cur_.merge(prev, on=["ticker", "year", "field"]).dropna(subset=["value", "value_prev"], how="all")

    units = selected.melt(["file", "ticker", "year"], ["unit_BS", "unit_IS", "unit_CF"], "statement", "unit")
    units["statement"] = units.statement.str.removeprefix("unit_")
    lines = facts.merge(units, on=["file", "statement"])
    lines["value"] = lines.cur * lines.unit.fillna(1.0)
    lines["value_prev"] = lines.prev * lines.unit.fillna(1.0)
    lines = lines[["ticker", "year", "statement", "code", "label", "value", "value_prev"]]

    names = feature_columns(panel) + ["m_score", "sgi", "tata", "market_rev_g"]
    metrics = panel.melt(["ticker", "year"], names, "name", "value").dropna(subset=["value"])

    raw = add_features(wide.set_index(["ticker", "year"])).reset_index()
    raw = raw.merge(panel[["ticker", "year", "usable", "sector"]], on=["ticker", "year"], how="left")
    risk = raw[raw.usable.eq(True) & raw.m_score.notna()]
    raw = raw.merge(
        risk.assign(
            m_pct_in_sector=risk.groupby(["sector", "year"]).m_score.rank(pct=True),
            accruals_pct_in_sector=risk.groupby(["sector", "year"]).accruals.rank(pct=True),
        )[["ticker", "year", "m_pct_in_sector", "accruals_pct_in_sector"]],
        on=["ticker", "year"], how="left",
    )
    ratio_names = [c for c in raw.columns if c not in ("ticker", "year", "usable", "sector")]
    ratios = raw.melt(["ticker", "year"], ratio_names, "name", "value").dropna(subset=["value"])

    forecasts = pd.DataFrame()
    pred_files = sorted((root / "artifacts").glob("predictions_*.csv"))
    if pred_files:
        f = pd.read_csv(pred_files[-1])
        f["target_year"] = f.year + 1
        forecasts = f[["ticker", "target_year", "pred_rel", "vol_group", "radius", "growth_lo", "growth_mid", "growth_hi"]]

    scope_path = root / "reference/scope_changes.csv"
    scope_changes = pd.read_csv(scope_path) if scope_path.exists() else pd.DataFrame()

    counts = {}
    with connect() as conn, conn.cursor() as cur:
        cur.execute(SCHEMA)
        for table, df in [("companies", companies), ("reports", reports), ("fundamentals", fundamentals),
                          ("statement_lines", lines), ("metrics", metrics), ("ratios", ratios), ("forecasts", forecasts),
                          ("scope_changes", scope_changes)]:
            counts[table] = _copy(cur, table, df) if len(df) else 0
    return counts


def to_vector(v) -> str:
    """pgvector literal for a 1-D array."""
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def store_document(ticker: str, year: int, chunks: list, embeddings, info: dict, doc_type: str = "annual_report") -> int:
    """Replace one report's chunks. `chunks` are finresearch.rag.ingest.Chunk objects."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(RAG_SCHEMA)
        cur.execute("DELETE FROM documents WHERE ticker = %s AND year = %s AND doc_type = %s", (ticker, year, doc_type))
        cur.execute(
            "INSERT INTO documents (ticker, year, doc_type, file, sha256, n_pages) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
            (ticker, year, doc_type, info["file"], info["sha256"], info["n_pages"]),
        )
        doc_id = cur.fetchone()[0]
        cur.executemany(
            "INSERT INTO chunks (id, document_id, ticker, year, page, section, kind, text, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)",
            [(c.id, doc_id, c.ticker, c.year, c.page, c.section, c.kind, c.text, to_vector(e)) for c, e in zip(chunks, embeddings)],
        )
    return len(chunks)
