"""Financial data tools: statements, ratios, peer comparison and the growth forecast.

    python -m finresearch.tools.financials FPT 2022 2025
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..db import query
from ..panel import FEATURE_LABELS
from .metrics import MetricSet

# field -> label
AMOUNTS = {
    "revenue": "Doanh thu thuần",
    "gross_profit": "Lợi nhuận gộp",
    "op_profit": "Lợi nhuận thuần từ hoạt động kinh doanh",
    "pbt": "Lợi nhuận trước thuế",
    "net_income": "Lợi nhuận sau thuế",
    "total_assets": "Tổng tài sản",
    "liabilities": "Nợ phải trả",
    "equity": "Vốn chủ sở hữu",
    "cash": "Tiền và tương đương tiền",
    "trade_receivables": "Phải thu khách hàng",
    "inventory": "Hàng tồn kho",
    "debt": "Vay và nợ thuê tài chính",
    "cfo": "Dòng tiền từ hoạt động kinh doanh",
    "capex": "Chi mua sắm tài sản cố định",
    "fcf": "Dòng tiền tự do",
    "dividends_paid": "Cổ tức đã trả",
}

# name -> (label, unit)
RATIOS = {
    "rev_g": ("Tăng trưởng doanh thu", "pct"),
    "net_income_g": ("Tăng trưởng lợi nhuận sau thuế", "pct"),
    "gross_margin": ("Biên lợi nhuận gộp", "pct"),
    "op_margin": ("Biên lợi nhuận hoạt động", "pct"),
    "net_margin": ("Biên lợi nhuận ròng", "pct"),
    "roe": ("ROE", "pct"),
    "roa": ("ROA", "pct"),
    "liab_to_assets": ("Nợ phải trả / tổng tài sản", "pct"),
    "debt_to_equity": ("Vay / vốn chủ sở hữu", "x"),
    "current_ratio": ("Hệ số thanh toán hiện hành", "x"),
    "asset_turnover": ("Vòng quay tài sản", "x"),
    "dso": ("Số ngày phải thu", "days"),
    "dio": ("Số ngày tồn kho", "days"),
    "cfo_to_net_income": ("Dòng tiền kinh doanh / lợi nhuận sau thuế", "x"),
    "fcf_margin": ("Dòng tiền tự do / doanh thu", "pct"),
    "accruals": ("Dồn tích / tổng tài sản", "pct"),
    "m_score": ("Beneish M-Score", "score"),
}
PEER_DEFAULT = ("rev_g", "gross_margin", "net_margin", "roe", "debt_to_equity", "dso", "accruals", "m_score")
SCOPE_LABEL = {"consolidated": "hợp nhất", "parent": "công ty mẹ", "single": "riêng lẻ"}


NO_HISTORY_EN = {"thiếu lịch sử": "no history"}


class UnknownTicker(ValueError):
    pass


def _t(lang: str, vi: str, en: str) -> str:
    """Notes are printed in the report as written, so they follow its language."""
    return en if lang == "en" else vi


@lru_cache(maxsize=1)
def _interval_coverage() -> float:
    path = Path(__file__).resolve().parents[2] / "artifacts/metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))["interval"]["coverage_target"] if path.exists() else 0.8


def _reports(ticker: str, start_year: int, end_year: int) -> pd.DataFrame:
    r = query("SELECT year, scope FROM reports WHERE ticker = %s AND year BETWEEN %s AND %s ORDER BY year",
              (ticker, start_year, end_year))
    if r.empty:
        if query("SELECT 1 FROM reports WHERE ticker = %s LIMIT 1", (ticker,)).empty:
            raise UnknownTicker(f"Không có báo cáo tài chính nào của {ticker} trong cơ sở dữ liệu")
    return r


def _source(year: int, scope: str) -> str:
    return f"BCTC kiểm toán {year} ({SCOPE_LABEL.get(scope, scope)})"


def _fundamentals(ticker: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Wide frame indexed by year: one column per field, plus `<field>_prev`."""
    f = query("SELECT year, field, value, value_prev FROM fundamentals WHERE ticker = %s AND year BETWEEN %s AND %s",
              (ticker, start_year, end_year))
    if f.empty:
        return pd.DataFrame()
    w = f.pivot(index="year", columns="field", values="value")
    wp = f.pivot(index="year", columns="field", values="value_prev").add_suffix("_prev")
    w = w.join(wp)
    for col in AMOUNTS:  # a field that no report had still needs its column
        for c in (col, col + "_prev"):
            if c not in w and col not in ("debt", "fcf"):
                w[c] = np.nan
    w["debt"] = w.get("st_debt", 0) + w.get("lt_debt", 0)
    w["fcf"] = w.cfo - w.capex
    return w.astype(float)


def get_financials(ticker: str, start_year: int, end_year: int, lang: str = "vi") -> MetricSet:
    """Key lines of the three statements, in VND."""
    ticker = ticker.upper()
    reports = _reports(ticker, start_year, end_year)
    w = _fundamentals(ticker, start_year, end_year)
    out = MetricSet(ticker=ticker)
    for year, scope in zip(reports.year, reports.scope):
        if year not in w.index:
            continue
        for field, label in AMOUNTS.items():
            out.add(field, year, w.at[year, field], "vnd", label, _source(year, scope))
    missing = sorted(set(range(start_year, end_year + 1)) - set(reports.year))
    if missing:
        years = ", ".join(map(str, missing))
        out.notes.append(_t(lang, f"Không có BCTC của {ticker} cho năm: {years}", f"No audited statements for {ticker} in: {years}"))
    return out


def _cagr(first: float, last: float, years: int) -> float | None:
    if years <= 0 or not (first > 0 and last > 0):
        return None
    return (last / first) ** (1 / years) - 1


def get_ratios(ticker: str, start_year: int, end_year: int, lang: str = "vi") -> MetricSet:
    """Growth, profitability, leverage, efficiency and cash-flow quality, plus CAGR over the period."""
    ticker = ticker.upper()
    reports = _reports(ticker, start_year, end_year)
    scope = dict(zip(reports.year, reports.scope))
    m = query("SELECT year, name, value FROM ratios WHERE ticker = %s AND year BETWEEN %s AND %s AND name = ANY(%s)",
              (ticker, start_year, end_year, list(RATIOS)))
    table = m.pivot(index="year", columns="name", values="value") if len(m) else pd.DataFrame()
    w = _fundamentals(ticker, start_year, end_year)
    if len(w):
        growth = w.net_income / w.net_income_prev.where(w.net_income_prev > 0) - 1
        table = table.join(growth.rename("net_income_g"), how="outer")
        table = table.join((w.cfo / w.net_income.where(w.net_income > 0)).rename("cfo_to_net_income"), how="outer")

    out = MetricSet(ticker=ticker)
    for year in sorted(table.index):
        for name, (label, unit) in RATIOS.items():
            if name in table:
                out.add(name, year, table.at[year, name], unit, label, f"Tính từ {_source(year, scope.get(year, ''))}")

    years = [y for y in sorted(w.index) if pd.notna(w.at[y, "revenue"])] if len(w) else []
    if len(years) >= 2:
        first, last = years[0], years[-1]
        period, src = f"{first}_{last}", f"Tính từ BCTC kiểm toán {first} và {last}"
        out.add("revenue_cagr", period, _cagr(w.at[first, "revenue"], w.at[last, "revenue"], last - first), "pct",
                f"CAGR doanh thu {first}–{last}", src)
        out.add("net_income_cagr", period, _cagr(w.at[first, "net_income"], w.at[last, "net_income"], last - first), "pct",
                f"CAGR lợi nhuận sau thuế {first}–{last}", src)
    return out


def get_peer_stats(ticker: str, year: int, names: tuple[str, ...] = PEER_DEFAULT, lang: str = "vi") -> MetricSet:
    """Sector median and the firm's percentile within its ICB sector for one year."""
    ticker = ticker.upper()
    sector = query("SELECT icb2 FROM companies WHERE ticker = %s", (ticker,))
    out = MetricSet(ticker=ticker)
    if sector.empty or sector.icb2.iloc[0] is None:
        out.notes.append(_t(lang, f"Không rõ ngành của {ticker} nên không so sánh được với công ty cùng ngành",
                            f"The sector of {ticker} is unknown, so there is no peer comparison"))
        return out
    sector = sector.icb2.iloc[0]
    peers = query(
        """SELECT m.ticker, m.name, m.value FROM ratios m
           JOIN companies c USING (ticker) JOIN reports r USING (ticker, year)
           WHERE c.icb2 = %s AND m.year = %s AND m.name = ANY(%s) AND r.usable""",
        (sector, year, list(names)),
    )
    for name in names:
        label, unit = RATIOS.get(name, (FEATURE_LABELS.get(name, name), "x"))
        group = peers[peers.name == name]
        own = group[group.ticker == ticker]
        if own.empty or len(group) < 5:
            continue
        src = f"{len(group)} công ty ngành {sector} có BCTC kiểm toán {year}"
        out.add(f"{name}_sector_median", year, group.value.median(), unit, f"{label}: trung vị ngành", src)
        out.add(f"{name}_sector_pct", year, (group.value <= own.value.iloc[0]).mean(), "pctile", f"{label}: vị trí trong ngành", src)
    n = peers.ticker.nunique()
    out.notes.append(_t(lang, f"So sánh với ngành {sector} ({n} công ty, năm {year})",
                        f"Peer comparison uses the ICB sector {sector} ({n} companies, {year})"))
    return out


def get_forecast(ticker: str, lang: str = "vi") -> MetricSet:
    """Model forecast of next year's revenue, with its interval and any change in consolidation scope."""
    ticker = ticker.upper()
    f = query("SELECT * FROM forecasts WHERE ticker = %s ORDER BY target_year DESC LIMIT 1", (ticker,))
    out = MetricSet(ticker=ticker)
    if f.empty:
        out.notes.append(_t(lang, f"Không có dự báo cho {ticker}: thiếu BCTC năm gần nhất hoặc không thuộc mẫu của mô hình",
                            f"No forecast for {ticker}: its latest statements are missing or it is outside the model's sample"))
        return out
    r = f.iloc[0]
    target, base_year = int(r.target_year), int(r.target_year) - 1
    base = _fundamentals(ticker, base_year, base_year).at[base_year, "revenue"]
    src = "Mô hình XGBoost trên BCTC kiểm toán 2015–2025; khoảng dự báo conformal 80%"

    changes = query("SELECT * FROM scope_changes WHERE ticker = %s AND effective_year = %s", (ticker, target))
    for c in changes.itertuples():
        out.notes.append(_t(lang, f"{c.note} (nguồn: {c.source})", f"{c.note_en} (source: {c.source})"))
        if c.kind == "deconsolidation":
            part = _fundamentals(c.counterpart, base_year, base_year)
            if len(part) and pd.notna(part.at[base_year, "revenue"]):
                removed = part.at[base_year, "revenue"]
                out.add("forecast_base_reported", base_year, base, "vnd", f"Doanh thu {base_year} theo báo cáo", _source(base_year, "consolidated"))
                out.add("forecast_base_removed", base_year, removed, "vnd", f"Doanh thu {base_year} của {c.counterpart}",
                        f"BCTC kiểm toán {base_year} của {c.counterpart}")
                base = base - removed
                out.notes.append(_t(
                    lang,
                    f"Doanh thu cơ sở {base_year} đã trừ doanh thu của {c.counterpart}. Đây là ước tính: chưa cộng lại giao dịch nội bộ "
                    f"giữa hai bên, và tốc độ tăng trưởng vẫn do mô hình tính trên số liệu còn gồm {c.counterpart}.",
                    f"The {base_year} revenue base excludes {c.counterpart}'s revenue. This is an estimate: intra-group sales are not "
                    f"added back, and the growth rate still comes from figures that include {c.counterpart}.",
                ))
            else:
                out.notes.append(_t(lang, f"Không có doanh thu {base_year} của {c.counterpart} để điều chỉnh; dự báo vẫn theo phạm vi hợp nhất cũ.",
                                    f"{c.counterpart}'s {base_year} revenue is not available, so the forecast keeps the old consolidation scope."))

    out.add("forecast_base", base_year, base, "vnd", f"Doanh thu cơ sở {base_year} cho dự báo", "Tính từ BCTC kiểm toán")
    for key, label in (("lo", "cận dưới"), ("mid", "điểm giữa"), ("hi", "cận trên")):
        g = r[f"growth_{key}"]
        out.add(f"forecast_growth_{key}", target, g, "pct", f"Tăng trưởng doanh thu dự báo {target}: {label}", src)
        out.add(f"forecast_revenue_{key}", target, base * (1 + g), "vnd", f"Doanh thu dự báo {target}: {label}", src)
    out.add("forecast_interval_coverage", target, _interval_coverage(), "pct", f"Mức phủ của khoảng dự báo {target}", src)
    out.notes.append(_t(
        lang,
        f"Dự báo giả định mặt bằng chung tăng như trung vị lịch sử; nhóm biến động doanh thu của {ticker}: {r.vol_group}.",
        f"The forecast assumes the market grows at its historical median; revenue volatility group of {ticker}: "
        f"{NO_HISTORY_EN.get(r.vol_group, r.vol_group)}.",
    ))
    return out


if __name__ == "__main__":
    tk, y0, y1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 60)
    for title, result in (("SỐ LIỆU", get_financials(tk, y0, y1)), ("CHỈ SỐ", get_ratios(tk, y0, y1)),
                          (f"SO VỚI NGÀNH {y1}", get_peer_stats(tk, y1)), ("DỰ BÁO", get_forecast(tk))):
        print(f"\n== {title} ==")
        print(result.frame().to_string())
        for note in result.notes:
            print("  *", note)
