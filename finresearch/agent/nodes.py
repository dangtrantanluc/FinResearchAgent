"""The steps before drafting: understand the question, pull the numbers, find the passages."""
from __future__ import annotations

import re

from ..db import query
from ..llm import LLM
from ..rag.search import Mode, search
from ..tools.financials import get_financials, get_forecast, get_peer_stats, get_ratios
from ..tools.risk import get_forecast_drivers, get_risk_signals
from .schema import Intent, State

DEFAULT_SPAN = 4  # years analysed when the question names none
VIETNAMESE = re.compile(r"[ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ]", re.I)


def _known_tickers() -> set[str]:
    return set(query("SELECT DISTINCT ticker FROM reports").ticker)


def parse_intent(question: str, llm: LLM | None = None) -> Intent | str:
    """Tickers and years by rule; the model is asked only when no ticker is written out.

    Returns an Intent, or a message for the user when the question cannot be served.
    """
    known = _known_tickers()
    tickers = [t for t in dict.fromkeys(re.findall(r"\b[A-Z][A-Z0-9]{2}\b", question)) if t in known]
    years = sorted({int(y) for y in re.findall(r"\b(20[0-3]\d)\b", question)})
    language = "vi" if VIETNAMESE.search(question) else "en"
    if not tickers and llm is not None:
        guess = llm.generate(Intent, "Extract the stock tickers (Vietnamese exchanges), the years and the language of the question. "
                                     "Use only tickers you are sure of; return an empty list otherwise.", question)
        tickers = [t.upper() for t in guess.tickers if t.upper() in known]
        years = years or [y for y in (guess.start_year, guess.end_year) if y]
    if not tickers:
        return ("Không nhận ra mã chứng khoán nào có trong cơ sở dữ liệu. Hãy viết mã bằng chữ in hoa, ví dụ: FPT, HPG."
                if language == "vi" else
                "No ticker in the question matches the database. Write tickers in capitals, for example FPT or HPG.")

    latest = int(query("SELECT max(year) AS y FROM reports WHERE ticker = ANY(%s)", (tickers,)).y.iloc[0])
    end = min(max(years), latest) if years else latest
    start = min(years) if len(years) >= 2 and min(years) < end else end - DEFAULT_SPAN + 1
    return Intent(tickers=tickers[:3], start_year=start, end_year=end,
                  kind="compare" if len(tickers) > 1 else "analyze", language=language)


def gather(state: State) -> dict:
    """Every number the report may use, each with an id. No model is involved."""
    intent = state["intent"]
    metrics, notes, flags = {}, [], []
    for ticker in intent.tickers:
        lang = intent.language
        results = [
            get_financials(ticker, intent.start_year, intent.end_year, lang=lang),
            get_ratios(ticker, intent.start_year, intent.end_year, lang=lang),
            get_peer_stats(ticker, intent.end_year, lang=lang),
            get_risk_signals(ticker, intent.end_year, lang=lang),
            get_forecast(ticker, lang=lang),
            get_forecast_drivers(ticker, lang=lang),
        ]
        for result in results:
            metrics.update({m.id: m for m in result.metrics})
            notes += [n for n in result.notes if n not in notes]
            flags += result.flags
    return {"metrics": metrics, "notes": notes, "flags": flags}


# topic -> question template; {t} ticker, {y} report year, {n} the year after
QUERY_TEMPLATES = {
    "revenue": ["Nguyên nhân tăng trưởng doanh thu của {t} năm {y}", "Kết quả kinh doanh theo từng khối, lĩnh vực và thị trường năm {y}"],
    "profitability": ["Biên lợi nhuận và hiệu quả hoạt động của {t} năm {y} thay đổi vì sao"],
    "cash_flow": ["Dòng tiền, đầu tư, vay nợ và cổ tức của {t} năm {y}"],
    "risk": ["Các rủi ro chính và biện pháp quản trị rủi ro của {t} năm {y}"],
    "forecast": ["Kế hoạch kinh doanh năm {n}: mục tiêu doanh thu, lợi nhuận và đầu tư", "Chiến lược phát triển và động lực tăng trưởng những năm tới của {t}"],
}


def plan_queries(state: State) -> dict:
    """Fixed questions per report section, asked of the latest report (and the one before for growth drivers)."""
    intent = state["intent"]
    queries, notes = [], list(state.get("notes", []))
    for ticker in intent.tickers:
        docs = set(query("SELECT year FROM documents WHERE ticker = %s", (ticker,)).year)
        years = [y for y in (intent.end_year, intent.end_year - 1) if y in docs and y >= intent.start_year]
        if not years:
            notes.append(
                f"The document store has no annual report of {ticker} for this period, so this note makes no qualitative statement about {ticker}."
                if intent.language == "en" else
                f"Kho tài liệu không có báo cáo thường niên của {ticker} cho giai đoạn này, nên báo cáo không có nhận định định tính về {ticker}.")
            continue
        for topic, templates in QUERY_TEMPLATES.items():
            for template in templates:
                for year in years if topic == "revenue" else years[:1]:
                    queries.append({"ticker": ticker, "year": year, "topic": topic,
                                    "text": template.format(t=ticker, y=year, n=year + 1)})
    return {"queries": queries, "notes": notes}


def retrieve(state: State, mode: Mode = "hybrid", per_query: int = 2) -> dict:
    """Two passages per question; nine questions per company keeps the draft prompt near 7k tokens."""
    evidence, topics = {}, {}
    for q in state.get("queries", []):
        for hit in search(q["text"], q["ticker"], [q["year"]], k=per_query, mode=mode):
            if hit.id not in evidence:
                evidence[hit.id] = hit
                topics[hit.id] = q["topic"]
    return {"evidence": evidence, "evidence_topics": topics}
