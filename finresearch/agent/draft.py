"""Write the report as claims. Two writers share one output shape.

`llm_draft` asks the model for prose. `template_draft` needs no model: it states
the numbers in fixed sentences, so the pipeline runs end to end without an API
key and tests have a deterministic writer.
"""
from __future__ import annotations

from ..llm import LLM
from ..tools.metrics import format_metric
from .schema import Claim, Report, Section, State

SYSTEM = """You are an equity research analyst writing about companies listed in Vietnam.
You write only from the material in the prompt. Rules, all mandatory:

1. Never type a figure that comes from METRICS. Write its id in double braces instead, e.g. {{FPT.revenue.2024}}.
   The system replaces the id with exactly the text in the value column, words included ("tỷ đồng", "lần", "phân vị",
   "điểm %", a minus sign), so do not write those words, a unit, a sign or "%" next to it. Because a negative value
   keeps its minus sign, introduce it with a neutral word ("ở mức", "là", "at", "of"), not with "giảm" or "fell".
2. A figure that is not in METRICS may appear only if it is printed in an EVIDENCE passage you cite, copied exactly
   as printed there (same digits and separators).
3. Any statement about causes, segments, markets, strategy, plans or risks must list the ids of the EVIDENCE passages
   that support it in evidence_ids. If no passage supports a statement, leave the statement out.
4. A claim that only states METRICS needs no evidence.
5. Years may be written as plain numbers.
   If the figure you need is not in METRICS, never stand in another metric for it: quote it from a passage under
   rule 2 or leave the statement out. Before writing that a figure rose or fell, compare the two values in METRICS.
6. No investment advice, no price targets, no adjectives the numbers do not justify.
7. Each claim is one or two sentences. Each section has two to five claims.
8. Sections, in this order: summary, revenue, profitability, cash_flow, risk, forecast.
   - risk: start from FLAGS (you may rephrase them but keep their {{ids}}); add risks the company itself describes in EVIDENCE.
   - forecast: give the model forecast with its lower and upper bound and the interval's coverage (it is a metric too),
     say what drives it using the driver_* metrics, and if EVIDENCE contains the company's own plan for that year,
     set the two side by side.
9. Write in {language}."""

LANGUAGE = {"vi": "Vietnamese", "en": "English"}


def build_prompt(state: State) -> str:
    intent = state["intent"]
    lang = intent.language
    lines = [f"QUESTION: {state['question']}", f"COMPANIES: {', '.join(intent.tickers)} | YEARS: {intent.start_year}-{intent.end_year}", "", "METRICS (id | meaning | value):"]
    lines += [f"{m.id} | {m.label} | {format_metric(m, lang)}" for m in state["metrics"].values()]
    lines += ["", "FLAGS:"] + ([f"- {f}" for f in state.get("flags", [])] or ["- none"])
    lines += ["", "NOTES (limits of the data; they are printed with the report, do not repeat them):"] + [f"- {n}" for n in state.get("notes", [])]
    lines += ["", "EVIDENCE (id | source | topic it was retrieved for):"]
    for eid, e in state.get("evidence", {}).items():
        lines += [f"[{eid}] {e.citation(lang)} | {state['evidence_topics'].get(eid, '')}", e.text, ""]
    return "\n".join(lines)


def llm_draft(state: State, llm: LLM) -> Report:
    system = SYSTEM.replace("{language}", LANGUAGE[state["intent"].language])
    return llm.generate(Report, system, build_prompt(state), tier="strong")


# Sentences for the offline writer. {r[name, period]} becomes a {{metric id}} reference.
TEMPLATES = {
    "vi": {
        "cagr": "Giai đoạn {s}–{e}, doanh thu thuần của {t} tăng trưởng kép {cagr_rev} mỗi năm và lợi nhuận sau thuế tăng trưởng kép {cagr_ni} mỗi năm.",
        "latest": "Năm {e}, {t} đạt doanh thu thuần {revenue} và lợi nhuận sau thuế {net_income}.",
        "revenue_year": "Năm {y}: doanh thu thuần {revenue}, thay đổi {rev_g} so với năm trước.",
        "margins": "Năm {y}: biên lợi nhuận gộp {gross_margin}, biên lợi nhuận ròng {net_margin}, ROE {roe}.",
        "peer": "So với ngành năm {e}: biên lợi nhuận ròng của {t} ở {net_margin_pct}, ROE ở {roe_pct}.",
        "cash": "Năm {y}: dòng tiền từ hoạt động kinh doanh {cfo}, bằng {cfo_to_ni} lợi nhuận sau thuế; dòng tiền tự do {fcf}.",
        "leverage": "Cuối năm {e}, vay và nợ thuê tài chính là {debt}, tương đương {debt_to_equity} vốn chủ sở hữu.",
        "no_flags": "Không có tín hiệu rủi ro kế toán nào vượt ngưỡng sàng lọc trong năm {e}; M-Score là {m_score}.",
        "forecast": "Mô hình dự báo doanh thu {n} của {t} là {fc_mid}, tương ứng tăng trưởng {g_mid}; khoảng dự báo với mức phủ {coverage} là từ {fc_lo} đến {fc_hi}.",
        "driver": "{label} làm dự báo thay đổi {value} so với mức trung bình.",
    },
    "en": {
        "cagr": "Over {s}–{e}, {t}'s net revenue compounded at {cagr_rev} a year and net profit at {cagr_ni} a year.",
        "latest": "In {e}, {t} reported net revenue of {revenue} and net profit of {net_income}.",
        "revenue_year": "{y}: net revenue {revenue}, a change of {rev_g} on the prior year.",
        "margins": "{y}: gross margin {gross_margin}, net margin {net_margin}, ROE {roe}.",
        "peer": "Against its sector in {e}: {t}'s net margin sits at {net_margin_pct} and its ROE at {roe_pct}.",
        "cash": "{y}: operating cash flow {cfo}, or {cfo_to_ni} of net profit; free cash flow {fcf}.",
        "leverage": "At the end of {e}, borrowings were {debt}, or {debt_to_equity} of equity.",
        "no_flags": "No accounting risk signal crossed its screening threshold in {e}; the M-Score is {m_score}.",
        "forecast": "The model forecasts {t}'s {n} revenue at {fc_mid}, growth of {g_mid}; the interval with {coverage} coverage runs from {fc_lo} to {fc_hi}.",
        "driver": "{label} moves the forecast by {value} against the average.",
    },
}


def template_draft(state: State) -> Report:
    """Numbers only, in fixed sentences. A sentence is skipped when one of its metrics is missing."""
    intent, metrics = state["intent"], state["metrics"]
    text = TEMPLATES[intent.language]
    sections: dict[str, list[Claim]] = {k: [] for k in ("summary", "revenue", "profitability", "cash_flow", "risk", "forecast")}

    def add(section: str, key: str, refs: dict[str, tuple[str, object]], **plain) -> None:
        ids = {name: f"{plain['t']}.{metric}.{period}" for name, (metric, period) in refs.items()}
        if all(i in metrics for i in ids.values()):
            sections[section].append(Claim(text=text[key].format(**plain, **{n: "{{" + i + "}}" for n, i in ids.items()})))

    for t in intent.tickers:
        s, e = intent.start_year, intent.end_year
        years = [y for y in range(s, e + 1) if f"{t}.revenue.{y}" in metrics]
        span = f"{years[0]}_{years[-1]}" if len(years) >= 2 else ""
        add("summary", "cagr", {"cagr_rev": ("revenue_cagr", span), "cagr_ni": ("net_income_cagr", span)}, t=t, s=years[0] if years else s, e=e)
        add("summary", "latest", {"revenue": ("revenue", e), "net_income": ("net_income", e)}, t=t, e=e)
        for y in years:
            add("revenue", "revenue_year", {"revenue": ("revenue", y), "rev_g": ("rev_g", y)}, t=t, y=y)
            add("profitability", "margins", {"gross_margin": ("gross_margin", y), "net_margin": ("net_margin", y), "roe": ("roe", y)}, t=t, y=y)
            add("cash_flow", "cash", {"cfo": ("cfo", y), "cfo_to_ni": ("cfo_to_net_income", y), "fcf": ("fcf", y)}, t=t, y=y)
        add("profitability", "peer", {"net_margin_pct": ("net_margin_sector_pct", e), "roe_pct": ("roe_sector_pct", e)}, t=t, e=e)
        add("cash_flow", "leverage", {"debt": ("debt", e), "debt_to_equity": ("debt_to_equity", e)}, t=t, e=e)
        own_flags = [f for f in state.get("flags", []) if "{{" + t + "." in f]
        sections["risk"] += [Claim(text=f) for f in own_flags]
        if not own_flags:
            add("risk", "no_flags", {"m_score": ("m_score", e)}, t=t, e=e)
        n = e + 1
        add("forecast", "forecast", {"fc_mid": ("forecast_revenue_mid", n), "fc_lo": ("forecast_revenue_lo", n),
                                     "fc_hi": ("forecast_revenue_hi", n), "g_mid": ("forecast_growth_mid", n),
                                     "coverage": ("forecast_interval_coverage", n)}, t=t, n=n)
        for m in metrics.values():
            if m.ticker == t and m.name.startswith("driver_"):
                label = m.label.split(": ", 1)[-1]
                sections["forecast"].append(Claim(text=text["driver"].format(label=label, value="{{" + m.id + "}}")))
    return Report(sections=[Section(key=k, claims=c) for k, c in sections.items() if c])
