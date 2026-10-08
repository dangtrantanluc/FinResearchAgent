"""Hybrid retrieval over report chunks.

    python -m finresearch.rag.search "Vì sao doanh thu FPT tăng năm 2024?" --ticker FPT --year 2024

Dense search (bge-m3, in Postgres) finds paraphrases and works across languages.
Lexical search (TF-IDF over syllable unigrams and bigrams, in memory) finds exact
names and terms. The two rankings are fused with reciprocal rank fusion. Optionally a
cross-encoder rereads the candidates against the question (mode "hybrid_rerank").

The lexical side is not Postgres full-text search on purpose: `ts_rank` has no
IDF, and Postgres ships no Vietnamese tokenizer.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

from ..db import query, to_vector
from . import models

Mode = Literal["dense", "lexical", "hybrid", "hybrid_rerank"]
CANDIDATES = 30
RRF_K = 60
SEARCHABLE = "kind <> 'numeric'"  # statement pages are skipped: their numbers come from the database


class Evidence(BaseModel):
    id: str  # e.g. "FPT-AR2024-p037-c1"
    ticker: str
    year: int
    page: int
    section: str
    text: str
    score: float

    def citation(self, lang: str = "vi") -> str:
        return f"BCTN {self.ticker} {self.year}, tr. {self.page}" if lang == "vi" else f"{self.ticker} Annual Report {self.year}, p. {self.page}"


@lru_cache(maxsize=2)
def _lexical_index(n_chunks: int):
    """TF-IDF matrix over all searchable chunks. Keyed on the chunk count so a new ingest rebuilds it."""
    df = query(f"SELECT id, ticker, year, text FROM chunks WHERE {SEARCHABLE} ORDER BY id")
    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    return df, vec, vec.fit_transform(df.text)


def _lexical(question: str, ticker: str | None, years: list[int] | None, n: int) -> list[str]:
    df, vec, matrix = _lexical_index(int(query(f"SELECT count(*) AS n FROM chunks WHERE {SEARCHABLE}").n.iloc[0]))
    scores = (matrix @ vec.transform([question]).T).toarray().ravel()
    mask = np.ones(len(df), dtype=bool)
    if ticker:
        mask &= (df.ticker == ticker).values
    if years:
        mask &= df.year.isin(years).values
    scores = np.where(mask & (scores > 0), scores, -1.0)
    top = np.argsort(-scores)[:n]
    return [df.id.iloc[i] for i in top if scores[i] > 0]


def _dense(question: str, ticker: str | None, years: list[int] | None, n: int) -> list[str]:
    vector = to_vector(models.embed([question])[0])
    sql = f"SELECT id FROM chunks WHERE {SEARCHABLE}"
    params: list = []
    if ticker:
        sql += " AND ticker = %s"
        params.append(ticker)
    if years:
        sql += " AND year = ANY(%s)"
        params.append(list(years))
    sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
    return query(sql, (*params, vector, n)).id.tolist()


def _fuse(rankings: list[list[str]]) -> dict[str, float]:
    """Reciprocal rank fusion: a chunk scores by its positions, not by incomparable raw scores."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    return scores


def search(question: str, ticker: str | None = None, years: list[int] | None = None, k: int = 6,
           mode: Mode = "hybrid") -> list[Evidence]:
    """Top-k chunks for a question. See eval/retrieval_results.json for how the modes compare:
    on the 20-question set hybrid found every answer in the top 6, and reranking moved more
    answers to rank 1 but pushed one out, so the cheaper mode is the default."""
    ticker = ticker.upper() if ticker else None
    rankings = []
    if mode != "lexical":
        rankings.append(_dense(question, ticker, years, CANDIDATES))
    if mode != "dense":
        rankings.append(_lexical(question, ticker, years, CANDIDATES))
    fused = _fuse(rankings)
    if not fused:
        return []
    ids = sorted(fused, key=fused.get, reverse=True)[:CANDIDATES]
    rows = query("SELECT id, ticker, year, page, section, text FROM chunks WHERE id = ANY(%s)", (ids,)).set_index("id").loc[ids]
    rows["score"] = [fused[i] for i in ids]
    if mode == "hybrid_rerank":
        passages = [f"{r.section}\n{r.text}" if r.section else r.text for r in rows.itertuples()]
        rows["score"] = models.rerank(question, passages).astype(float)
        rows = rows.sort_values("score", ascending=False)
    return [Evidence(id=i, **r) for i, r in rows.head(k).to_dict("index").items()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--ticker")
    ap.add_argument("--year", type=int, action="append")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--mode", default="hybrid", choices=["dense", "lexical", "hybrid", "hybrid_rerank"])
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    for n, e in enumerate(search(args.question, args.ticker, args.year, args.k, args.mode), 1):
        print(f"\n[{n}] {e.id} | {e.citation()} | điểm {e.score:.3f} | {e.section}")
        print("    " + e.text[:420].replace("\n", " ") + "…")
