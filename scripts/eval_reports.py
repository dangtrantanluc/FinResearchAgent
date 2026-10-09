"""Run a fixed set of questions through the full workflow and record what the validator did.

    python scripts/eval_reports.py [question id ...]

Writes eval/report_results.json and one markdown report per question under eval/reports/.
With ids, only those questions are run and the other rows of the results file are kept.
Uses the configured language model; without a key it measures the offline mode.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finresearch.agent.graph import build_graph  # noqa: E402
from finresearch.env import load_env  # noqa: E402
from finresearch.llm import default_llm  # noqa: E402

QUESTIONS = {
    "fpt-vi": "Phân tích kết quả kinh doanh của FPT giai đoạn 2022–2025, chỉ ra động lực tăng trưởng, rủi ro tài chính và ước tính doanh thu 2026",
    "fpt-en": "Analyze FPT's financial performance from 2022–2025, identify the main drivers of growth, detect potential financial risks, and estimate 2026 revenue.",
    "hpg-vi": "Đánh giá sức khỏe tài chính của HPG giai đoạn 2023–2025",
    "mwg-en": "Should I be concerned about MWG's financial performance between 2022 and 2025?",
    "fpt-elc": "So sánh FPT và ELC giai đoạn 2023–2025",
}

if __name__ == "__main__":
    load_env(ROOT)
    llm = default_llm()
    graph = build_graph(llm)
    out_dir = ROOT / "eval/reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = ROOT / "eval/report_results.json"
    only = sys.argv[1:]
    previous = json.loads(results_path.read_text(encoding="utf-8")) if only and results_path.exists() else {"summary": [], "claims": {}}
    rows = [r for r in previous["summary"] if r["id"] not in only]
    details = {k: v for k, v in previous["claims"].items() if k not in only}
    for key, question in QUESTIONS.items():
        if only and key not in only:
            continue
        started = time.time()
        try:
            final = graph.invoke({"question": question})
        except Exception as exc:  # a quota error on one question should not lose the others
            rows.append({"id": key, "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
            continue
        (out_dir / f"{key}.md").write_text(final.get("markdown") or final.get("error", ""), encoding="utf-8")
        answered = [c for c in final.get("llm_calls", []) if "error" not in c]
        rows.append({"id": key, **final.get("stats", {}), "evidence": len(final.get("evidence", {})),
                     "seconds": round(time.time() - started), "models": ", ".join(sorted({c["model"] for c in answered})),
                     "failed_calls": sum("error" in c for c in final.get("llm_calls", []))})
        details[key] = [{"section": c.section, "text": c.claim.text, "kept": c.kept, "support": c.support,
                         "problems": c.problems, "reason": c.reason} for c in final.get("checked", [])]
    rows.sort(key=lambda r: list(QUESTIONS).index(r["id"]))
    table = pd.DataFrame(rows).set_index("id")
    print("mode:", "Gemini" if llm else "offline (no API key)")
    print(table.to_string())
    results_path.write_text(
        json.dumps({"mode": "llm" if llm else "offline", "summary": rows, "claims": details}, ensure_ascii=False, indent=2), encoding="utf-8")
