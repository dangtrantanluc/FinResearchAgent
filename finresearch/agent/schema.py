"""Shapes passed between the steps of the workflow."""
from __future__ import annotations

import re
from typing import Literal, TypedDict

from pydantic import BaseModel, Field

from ..rag.search import Evidence
from ..tools.metrics import Metric

SectionKey = Literal["summary", "revenue", "profitability", "cash_flow", "risk", "forecast"]
SECTION_TITLES = {
    "summary": ("Tóm tắt", "Executive summary"),
    "revenue": ("Doanh thu", "Revenue"),
    "profitability": ("Lợi nhuận", "Profitability"),
    "cash_flow": ("Dòng tiền và cấu trúc vốn", "Cash flow and capital structure"),
    "risk": ("Tín hiệu rủi ro", "Risk signals"),
    "forecast": ("Dự báo", "Forecast"),
}
PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_.]+)\s*\}\}")


class Intent(BaseModel):
    tickers: list[str] = Field(description="Stock tickers the question is about, upper case")
    start_year: int | None = Field(None, description="First fiscal year to analyse, if the question names one")
    end_year: int | None = Field(None, description="Last fiscal year with reported results to analyse")
    kind: Literal["analyze", "compare"] = "analyze"
    language: Literal["vi", "en"] = "vi"


class Claim(BaseModel):
    text: str = Field(description="One or two sentences. Figures from METRICS appear only as {{metric id}}.")
    evidence_ids: list[str] = Field(default_factory=list, description="Ids of the EVIDENCE passages that support this claim")

    def metric_ids(self) -> list[str]:
        return PLACEHOLDER.findall(self.text)


class Section(BaseModel):
    key: SectionKey
    claims: list[Claim]


class Report(BaseModel):
    sections: list[Section]


class Verdict(BaseModel):
    index: int = Field(description="Number of the claim being judged")
    verdict: Literal["supported", "partial", "unsupported"]
    reason: str = Field(description="One short sentence")


class Verdicts(BaseModel):
    items: list[Verdict]


class CheckedClaim(BaseModel):
    section: SectionKey
    claim: Claim
    problems: list[str] = []                 # deterministic failures; any of these drops the claim
    support: Literal["supported", "partial", "unsupported", "unchecked", "none_needed"] = "none_needed"
    reason: str = ""

    @property
    def kept(self) -> bool:
        return not self.problems and self.support != "unsupported"


class State(TypedDict, total=False):
    question: str
    intent: Intent
    error: str
    metrics: dict[str, Metric]
    notes: list[str]
    flags: list[str]
    queries: list[dict]                      # {"ticker", "year", "topic", "text"}
    evidence: dict[str, Evidence]
    evidence_topics: dict[str, str]
    draft: Report
    checked: list[CheckedClaim]
    markdown: str
    stats: dict
    llm_calls: list[dict]                    # every model attempt of the run, failed ones included
