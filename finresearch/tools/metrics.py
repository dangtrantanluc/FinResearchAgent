"""A number the report may cite, and how to print it."""
from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel

Unit = Literal["vnd", "pct", "pp", "x", "days", "score", "pctile"]


class Metric(BaseModel):
    id: str  # "<ticker>.<name>.<period>", e.g. "FPT.revenue.2024"
    ticker: str
    name: str
    period: str
    value: float
    unit: Unit
    label: str
    source: str


class MetricSet(BaseModel):
    ticker: str
    metrics: list[Metric] = []
    notes: list[str] = []  # caveats the report must carry, e.g. a change in consolidation scope
    flags: list[str] = []  # ready-made findings; numbers appear only as {{metric id}} references

    def add(self, name: str, period, value, unit: Unit, label: str, source: str) -> None:
        if value is None or pd.isna(value):
            return
        self.metrics.append(Metric(id=f"{self.ticker}.{name}.{period}", ticker=self.ticker, name=name, period=str(period),
                                   value=float(value), unit=unit, label=label, source=source))

    def get(self, name: str, period) -> Metric | None:
        return next((m for m in self.metrics if m.name == name and m.period == str(period)), None)

    def frame(self, lang: str = "vi") -> pd.DataFrame:
        """Rows = metrics, columns = periods, cells formatted for reading."""
        rows: dict[str, dict[str, str]] = {}
        for m in self.metrics:
            rows.setdefault(m.label, {})[m.period] = format_metric(m, lang)
        return pd.DataFrame(rows).T.fillna("")


def _num(x: float, decimals: int, lang: str) -> str:
    s = f"{x:,.{decimals}f}"
    return s.translate(str.maketrans(",.", ".,")) if lang == "vi" else s


def format_metric(m: Metric, lang: str = "vi") -> str:
    v = m.value
    if m.unit == "vnd":
        return f"{_num(v / 1e9, 0, lang)} tỷ đồng" if lang == "vi" else f"VND {_num(v / 1e9, 0, lang)} bn"
    if m.unit == "pct":
        return f"{_num(v * 100, 1, lang)}%"
    if m.unit == "pp":  # a difference between two percentages
        sign = "+" if v > 0 else ""
        return f"{sign}{_num(v * 100, 1, lang)} điểm %" if lang == "vi" else f"{sign}{_num(v * 100, 1, lang)} pp"
    if m.unit == "x":
        return f"{_num(v, 2, lang)} lần" if lang == "vi" else f"{_num(v, 2, lang)}x"
    if m.unit == "days":
        return f"{_num(v, 0, lang)} ngày" if lang == "vi" else f"{_num(v, 0, lang)} days"
    if m.unit == "pctile":
        return f"phân vị {_num(v * 100, 0, lang)}" if lang == "vi" else f"percentile {_num(v * 100, 0, lang)}"
    return _num(v, 2, lang)
