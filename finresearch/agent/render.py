"""Turn the surviving claims into the report the reader sees."""
from __future__ import annotations

from ..tools.metrics import format_metric
from .schema import PLACEHOLDER, SECTION_TITLES, State

EXCERPTS_PER_TOPIC = 2
EXCERPT_CHARS = 320

# Rows of the side-by-side table: (metric name, Vietnamese label, English label, show the sector percentile?)
COMPARISON_ROWS = [
    ("revenue", "Doanh thu thuần {e}", "Net revenue {e}", False),
    ("revenue_cagr", "CAGR doanh thu {s}–{e}", "Revenue CAGR {s}–{e}", False),
    ("rev_g", "Tăng trưởng doanh thu {e}", "Revenue growth {e}", True),
    ("gross_margin", "Biên lợi nhuận gộp {e}", "Gross margin {e}", True),
    ("net_margin", "Biên lợi nhuận ròng {e}", "Net margin {e}", True),
    ("roe", "ROE {e}", "ROE {e}", True),
    ("debt_to_equity", "Vay / vốn chủ sở hữu {e}", "Debt / equity {e}", True),
    ("cfo_to_net_income", "Dòng tiền kinh doanh / LNST {e}", "Operating cash flow / net profit {e}", False),
    ("fcf_margin", "Dòng tiền tự do / doanh thu {e}", "Free cash flow / revenue {e}", False),
    ("dso", "Số ngày phải thu {e}", "Days sales outstanding {e}", True),
    ("m_score", "Beneish M-Score {e}", "Beneish M-Score {e}", True),
    ("forecast_growth_mid", "Tăng trưởng dự báo {n}", "Forecast growth {n}", False),
    ("forecast_revenue_mid", "Doanh thu dự báo {n}", "Forecast revenue {n}", False),
]


def comparison_table(state: State) -> list[str]:
    """Key figures of each company next to each other. Built from the metrics, no model involved."""
    intent, metrics = state["intent"], state["metrics"]
    lang, e = intent.language, intent.end_year
    by_name: dict[tuple[str, str], list] = {}
    for m in metrics.values():
        by_name.setdefault((m.ticker, m.name), []).append(m)

    def cell(ticker: str, name: str, with_pct: bool) -> str:
        found = by_name.get((ticker, name), [])
        # the latest period on record: the end year for annual figures, the whole span for a CAGR
        m = max(found, key=lambda x: x.period, default=None) if name == "revenue_cagr" else \
            next((x for x in found if x.period in (str(e), str(e + 1))), None)
        if m is None:
            return "—"
        text = format_metric(m, lang)
        if name.startswith("forecast_"):
            lo, hi = (metrics.get(m.id.replace("_mid.", f"_{k}.")) for k in ("lo", "hi"))
            if lo and hi:
                text += f" ({format_metric(lo, lang)} – {format_metric(hi, lang)})"
        pct = metrics.get(f"{ticker}.{name}_sector_pct.{e}") if with_pct else None
        return f"{text} ({format_metric(pct, lang)})" if pct else text

    head = "Chỉ tiêu" if lang == "vi" else "Metric"
    lines = [f"| {head} | " + " | ".join(intent.tickers) + " |", "|---|" + "---|" * len(intent.tickers)]
    for name, vi, en, with_pct in COMPARISON_ROWS:
        cells = [cell(t, name, with_pct) for t in intent.tickers]
        if any(c != "—" for c in cells):
            lines.append(f"| {(vi if lang == 'vi' else en).format(s=intent.start_year, e=e, n=e + 1)} | " + " | ".join(cells) + " |")
    note = ("Trong ngoặc là vị trí của công ty trong ngành của chính nó (phân vị 100 là cao nhất); với dự báo, trong ngoặc là khoảng dự báo."
            if lang == "vi" else
            "Brackets give the company's position within its own sector (percentile 100 is the highest); for forecasts they give the interval.")
    return lines + ["", note, ""]

TEXT = {
    "vi": {
        "title": "Báo cáo nghiên cứu: {tickers} ({s}–{e})",
        "limits": "Giới hạn và lưu ý",
        "side_by_side": "So sánh nhanh",
        "excerpts": "Đoạn trích liên quan (chưa được tổng hợp)",
        "excerpts_why": "Chưa cấu hình mô hình ngôn ngữ nên các đoạn dưới đây được trích nguyên văn từ báo cáo thường niên, chưa viết thành nhận định.",
        "confidence": "Độ tin cậy",
        "sources": "Nguồn",
        "numbers": "Số liệu tài chính: BCTC kiểm toán {s}–{e}, trích từ bộ dữ liệu OCR và đã kiểm tra bằng đẳng thức kế toán.",
        "conf_numbers": "- **Số liệu:** cao. Mọi con số trong báo cáo được điền từ cơ sở dữ liệu, không do mô hình ngôn ngữ viết.",
        "conf_judged": "- **Nhận định định tính:** {supported}/{qualitative} nhận định được đoạn trích hỗ trợ đầy đủ, {partial} hỗ trợ một phần.",
        "conf_unchecked": "- **Nhận định định tính:** {qualitative} nhận định có trích dẫn nhưng chưa được kiểm tra bằng mô hình ngôn ngữ.",
        "conf_none": "- **Nhận định định tính:** không có; báo cáo chỉ gồm số liệu.",
        "conf_forecast": "- **Dự báo:** thấp. Mô hình chỉ hơn baseline khoảng 3% trên dữ liệu kiểm tra; hãy đọc khoảng dự báo, đừng đọc điểm giữa.",
        "dropped": "- **Đã loại:** {n} nhận định không qua kiểm tra.",
        "disclaimer": "*Báo cáo này phục vụ nghiên cứu, không phải khuyến nghị đầu tư.*",
    },
    "en": {
        "title": "Research note: {tickers} ({s}–{e})",
        "limits": "Limits and caveats",
        "side_by_side": "Side by side",
        "excerpts": "Related passages (not synthesised)",
        "excerpts_why": "No language model is configured, so the passages below are quoted from the annual reports as found, not turned into statements.",
        "confidence": "Confidence",
        "sources": "Sources",
        "numbers": "Financial figures: audited statements {s}–{e}, extracted from an OCR dataset and checked against accounting identities.",
        "conf_numbers": "- **Figures:** high. Every number is filled in from the database, none is written by the language model.",
        "conf_judged": "- **Qualitative statements:** {supported} of {qualitative} are fully supported by their passages, {partial} partly.",
        "conf_unchecked": "- **Qualitative statements:** {qualitative} carry citations but were not checked by a language model.",
        "conf_none": "- **Qualitative statements:** none; this note contains figures only.",
        "conf_forecast": "- **Forecast:** low. The model beats its baseline by about 3% on held-out data; read the interval, not the midpoint.",
        "dropped": "- **Removed:** {n} statements that failed validation.",
        "disclaimer": "*This note is for research and is not investment advice.*",
    },
}


def render(state: State) -> dict:
    intent, metrics, evidence = state["intent"], state["metrics"], state.get("evidence", {})
    lang = intent.language
    t = TEXT[lang]
    numbering: dict[str, int] = {}

    def fill(claim) -> str:
        body = PLACEHOLDER.sub(lambda m: format_metric(metrics[m.group(1)], lang), claim.text)
        marks = "".join(f"[{numbering.setdefault(e, len(numbering) + 1)}]" for e in claim.evidence_ids)
        return f"{body} {marks}".strip()

    out = ["# " + t["title"].format(tickers=", ".join(intent.tickers), s=intent.start_year, e=intent.end_year), ""]
    if len(intent.tickers) > 1:
        out += [f"## {t['side_by_side']}", ""] + comparison_table(state)
    kept = [c for c in state["checked"] if c.kept]
    for key, titles in SECTION_TITLES.items():
        claims = [c for c in kept if c.section == key]
        if claims:
            out += [f"## {titles[lang == 'en']}", ""] + [f"- {fill(c.claim)}" for c in claims] + [""]

    if evidence and not numbering:  # offline mode: show what retrieval found instead of hiding it
        out += [f"## {t['excerpts']}", "", t["excerpts_why"], ""]
        shown: dict[str, int] = {}
        for eid, e in evidence.items():
            topic = state.get("evidence_topics", {}).get(eid, "")
            if shown.get(topic, 0) < EXCERPTS_PER_TOPIC:
                shown[topic] = shown.get(topic, 0) + 1
                excerpt = " ".join(e.text.split())[:EXCERPT_CHARS]
                out.append(f"- **{e.citation(lang)}** ({SECTION_TITLES.get(topic, (topic, topic))[lang == 'en']}): {excerpt}…")
        out.append("")

    if state.get("notes"):
        out += [f"## {t['limits']}", ""] + [f"- {n}" for n in state["notes"]] + [""]

    s = state["stats"]
    out += [f"## {t['confidence']}", "", t["conf_numbers"]]
    if s["qualitative"] == 0:
        out.append(t["conf_none"])
    elif s["unchecked"] == s["qualitative"]:
        out.append(t["conf_unchecked"].format(**s))
    else:
        out.append(t["conf_judged"].format(**s))
    if any(c.section == "forecast" for c in kept):
        out.append(t["conf_forecast"])
    if s["claims"] - s["kept"]:
        out.append(t["dropped"].format(n=s["claims"] - s["kept"]))

    out += ["", f"## {t['sources']}", "", "- " + t["numbers"].format(s=intent.start_year, e=intent.end_year)]
    out += [f"- [{n}] {evidence[e].citation(lang)}" for e, n in numbering.items()]
    out += ["", t["disclaimer"]]
    return {"markdown": "\n".join(out)}
