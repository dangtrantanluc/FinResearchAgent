"""Rule-based accounting risk signals, each tied to the numbers that triggered it."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..db import query
from ..panel import FEATURE_LABELS, FEATURE_LABELS_EN
from .metrics import MetricSet

ROOT = Path(__file__).resolve().parents[2]

# name -> ((Vietnamese label, English label), unit)
SIGNALS = {
    "m_score": (("Beneish M-Score", "Beneish M-Score"), "score"),
    "m_pct_in_sector": (("Vị trí M-Score trong ngành", "M-Score position in sector"), "pctile"),
    "accruals": (("Dồn tích / tổng tài sản", "Accruals / total assets"), "pct"),
    "accruals_pct_in_sector": (("Vị trí dồn tích trong ngành", "Accruals position in sector"), "pctile"),
    "recv_minus_rev_g": (("Tăng trưởng phải thu trừ tăng trưởng doanh thu", "Receivables growth minus revenue growth"), "pp"),
    "dso_chg": (("Thay đổi số ngày phải thu", "Change in days sales outstanding"), "days"),
    "debt_g": (("Tăng trưởng vay", "Growth in borrowings"), "pct"),
    "debt_to_equity": (("Vay / vốn chủ sở hữu", "Debt / equity"), "x"),
}


def get_risk_signals(ticker: str, year: int, lang: str = "vi") -> MetricSet:
    """Risk metrics for one year plus the flags that fire. Thresholds are screening rules, not verdicts."""
    ticker = ticker.upper()
    out = MetricSet(ticker=ticker)
    m = query("SELECT name, value FROM ratios WHERE ticker = %s AND year = %s AND name = ANY(%s)", (ticker, year, list(SIGNALS)))
    values = dict(zip(m.name, m.value))
    f = query("SELECT field, value FROM fundamentals WHERE ticker = %s AND year = %s AND field IN ('cfo', 'net_income')", (ticker, year))
    base = dict(zip(f.field, f.value))
    src = f"Computed from audited statements {year}" if lang == "en" else f"Tính từ BCTC kiểm toán {year}"
    for name, (label, unit) in SIGNALS.items():
        out.add(name, year, values.get(name), unit, label[lang == "en"], src)
    if base.get("net_income", 0) and base["net_income"] > 0 and base.get("cfo") is not None:
        values["cfo_to_net_income"] = base["cfo"] / base["net_income"]
        out.add("cfo_to_net_income", year, values["cfo_to_net_income"], "x",
                "Operating cash flow / net profit" if lang == "en" else "Dòng tiền kinh doanh / lợi nhuận sau thuế", src)

    ref = lambda name: "{{" + f"{ticker}.{name}.{year}" + "}}"
    v = lambda name: values.get(name, np.nan)
    t = lambda vi, en: en if lang == "en" else vi
    if v("m_pct_in_sector") >= 0.8:
        out.flags.append(t(f"M-Score {ref('m_score')} thuộc nhóm cao nhất ngành ({ref('m_pct_in_sector')}), nên xem kỹ chất lượng lợi nhuận.",
                           f"The M-Score of {ref('m_score')} is among the highest in the sector ({ref('m_pct_in_sector')}); earnings quality deserves a closer look."))
    if v("recv_minus_rev_g") > 0.15:
        out.flags.append(t(f"Phải thu khách hàng tăng nhanh hơn doanh thu {ref('recv_minus_rev_g')}; số ngày phải thu thay đổi {ref('dso_chg')}.",
                           f"Trade receivables grew faster than revenue by {ref('recv_minus_rev_g')}; days sales outstanding changed by {ref('dso_chg')}."))
    if v("accruals_pct_in_sector") >= 0.8 and v("accruals") > 0:
        out.flags.append(t(f"Lợi nhuận vượt dòng tiền kinh doanh: dồn tích bằng {ref('accruals')} tổng tài sản, cao so với ngành ({ref('accruals_pct_in_sector')}).",
                           f"Profit runs ahead of operating cash flow: accruals equal {ref('accruals')} of total assets, high for the sector ({ref('accruals_pct_in_sector')})."))
    if v("cfo_to_net_income") < 0.5:
        out.flags.append(t(f"Dòng tiền kinh doanh chỉ bằng {ref('cfo_to_net_income')} lợi nhuận sau thuế.",
                           f"Operating cash flow is only {ref('cfo_to_net_income')} of net profit."))
    if v("debt_g") > 0.3:
        out.flags.append(t(f"Vay và nợ thuê tài chính tăng {ref('debt_g')} trong năm; vay trên vốn chủ sở hữu ở mức {ref('debt_to_equity')}.",
                           f"Borrowings rose {ref('debt_g')} during the year; debt stands at {ref('debt_to_equity')} of equity."))
    if not out.metrics:
        out.notes.append(t(f"Không đủ số liệu năm {year} để tính tín hiệu rủi ro cho {ticker}", f"Not enough {year} data to compute risk signals for {ticker}"))
    elif not out.flags:
        out.notes.append(t(f"Không có tín hiệu rủi ro nào vượt ngưỡng sàng lọc trong năm {year}", f"No risk signal crossed its screening threshold in {year}"))
    return out


@lru_cache(maxsize=1)
def _model():
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(ROOT / "artifacts/growth_model.json"))
    features = json.loads((ROOT / "artifacts/metrics.json").read_text(encoding="utf-8"))["features"]
    return booster, features


def get_forecast_drivers(ticker: str, top: int = 5, lang: str = "vi") -> MetricSet:
    """SHAP contributions behind the forecast: which of the firm's own numbers push it up or down."""
    import xgboost as xgb

    ticker = ticker.upper()
    out = MetricSet(ticker=ticker)
    f = query("SELECT target_year, pred_rel FROM forecasts WHERE ticker = %s ORDER BY target_year DESC LIMIT 1", (ticker,))
    if f.empty or not (ROOT / "artifacts/growth_model.json").exists():
        return out
    target, base_year = int(f.target_year.iloc[0]), int(f.target_year.iloc[0]) - 1
    booster, features = _model()
    m = query("SELECT name, value FROM metrics WHERE ticker = %s AND year = %s", (ticker, base_year))
    row = pd.DataFrame([dict(zip(m.name, m.value))]).reindex(columns=features).astype(float)
    contrib = booster.predict(xgb.DMatrix(row), pred_contribs=True)[0]
    if abs(contrib.sum() - f.pred_rel.iloc[0]) > 1e-3:  # the database and the model file must describe the same model
        out.notes.append("Giải thích SHAP không khớp dự báo đã lưu; hãy nạp lại cơ sở dữ liệu sau khi train lại." if lang != "en" else
                         "The SHAP explanation does not match the stored forecast; reload the database after retraining.")
        return out
    src = "TreeSHAP of the growth model (artifacts/growth_model.json)" if lang == "en" else "TreeSHAP của mô hình tăng trưởng (artifacts/growth_model.json)"
    out.add("forecast_rel", target, f.pred_rel.iloc[0], "pp",
            f"Forecast {target}: gap to the market median" if lang == "en" else f"Dự báo {target}: lệch so với mặt bằng chung", src)
    for i in np.argsort(-np.abs(contrib[:-1]))[:top]:
        name = features[i]
        labels, prefix = (FEATURE_LABELS_EN, "Contribution to the forecast") if lang == "en" else (FEATURE_LABELS, "Đóng góp vào dự báo")
        out.add(f"driver_{name}", target, contrib[i], "pp", f"{prefix}: {labels.get(name, name)}", src)
    return out
