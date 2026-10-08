"""Score the retriever on eval/retrieval_fpt.jsonl.

A question is a hit when one of the top-k chunks comes from a gold page.
Each search is restricted to the question's ticker and report year, as the
agent does when it knows which report it is asking about.

    python scripts/eval_retrieval.py [--k 6]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finresearch.rag.search import search  # noqa: E402

MODES = ["lexical", "dense", "hybrid", "hybrid_rerank"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()
    questions = [json.loads(l) for l in (ROOT / "eval/retrieval_fpt.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    rows = []
    for q in questions:
        for mode in MODES:
            hits = search(q["question"], q["ticker"], [q["year"]], k=args.k, mode=mode)
            rank = next((n for n, e in enumerate(hits, 1) if e.page in q["gold_pages"]), None)
            rows.append({"id": q["id"], "lang": q["lang"], "mode": mode, "rank": rank, "pages": [e.page for e in hits]})
    df = pd.DataFrame(rows)
    df["hit"] = df["rank"].notna()
    df["rr"] = (1 / df["rank"]).fillna(0.0)

    summary = df.groupby("mode").agg(recall=("hit", "mean"), mrr=("rr", "mean")).reindex(MODES)
    by_lang = df.pivot_table(index="mode", columns="lang", values="hit", aggfunc="mean").reindex(MODES)
    summary = summary.join(by_lang.add_prefix("recall_"))
    print(f"{len(questions)} câu hỏi, k = {args.k}")
    print(summary.round(3).to_string())
    missed = df[(df["mode"] == "hybrid_rerank") & ~df.hit]
    for r in missed.itertuples():
        q = next(q for q in questions if q["id"] == r.id)
        print(f"  trượt {r.id} ({q['lang']}): {q['question']} | gold {q['gold_pages']} | trả về {r.pages}")

    out = {"k": args.k, "n_questions": len(questions), "summary": summary.round(4).reset_index().to_dict("records"),
           "per_question": df.drop(columns=["rr"]).to_dict("records")}
    (ROOT / "eval/retrieval_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
