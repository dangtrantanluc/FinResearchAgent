import pytest

from finresearch.tools.financials import UnknownTicker, get_financials, get_forecast, get_peer_stats, get_ratios
from finresearch.tools.metrics import Metric, format_metric


def _db_ready() -> bool:
    try:
        from finresearch.db import query
        return not query("SELECT 1 FROM reports LIMIT 1").empty
    except Exception:
        return False


needs_db = pytest.mark.skipif(not _db_ready(), reason="Postgres not running or not loaded (scripts/load_db.py)")


def _metric(value, unit):
    return Metric(id="X.m.2024", ticker="X", name="m", period="2024", value=value, unit=unit, label="m", source="s")


@pytest.mark.parametrize("value, unit, vi, en", [
    (62_848_794_230_018, "vnd", "62.849 tỷ đồng", "VND 62,849 bn"),
    (0.1943, "pct", "19,4%", "19.4%"),
    (0.876, "x", "0,88 lần", "0.88x"),
    (61.2, "days", "61 ngày", "61 days"),
    (-2.46, "score", "-2,46", "-2.46"),
])
def test_format_metric(value, unit, vi, en):
    assert format_metric(_metric(value, unit), "vi") == vi
    assert format_metric(_metric(value, unit), "en") == en


@needs_db
def test_fpt_revenue_matches_audited_figures():
    fin = get_financials("fpt", 2022, 2024)
    assert [round(fin.get("revenue", y).value / 1e9) for y in (2022, 2023, 2024)] == [44010, 52618, 62849]
    assert fin.get("revenue", 2024).id == "FPT.revenue.2024"
    assert "hợp nhất" in fin.get("revenue", 2024).source


@needs_db
def test_metric_ids_are_unique():
    ids = [m.id for result in (get_financials("FPT", 2022, 2025), get_ratios("FPT", 2022, 2025), get_peer_stats("FPT", 2025))
           for m in result.metrics]
    assert len(ids) == len(set(ids))


@needs_db
def test_cagr_is_computed_from_endpoints():
    ratios = get_ratios("FPT", 2022, 2025)
    fin = get_financials("FPT", 2022, 2025)
    expected = (fin.get("revenue", 2025).value / fin.get("revenue", 2022).value) ** (1 / 3) - 1
    assert ratios.get("revenue_cagr", "2022_2025").value == pytest.approx(expected)
    assert ratios.get("gross_margin", 2024).value == pytest.approx(fin.get("gross_profit", 2024).value / fin.get("revenue", 2024).value)


@needs_db
def test_missing_years_are_reported_not_invented():
    fin = get_financials("FPT", 2012, 2016)
    assert fin.get("revenue", 2013) is None and fin.get("revenue", 2016) is not None
    assert any("2012" in n and "2014" in n for n in fin.notes)


@needs_db
def test_unknown_ticker_raises():
    with pytest.raises(UnknownTicker):
        get_financials("ZZZZ", 2022, 2025)


@needs_db
def test_peer_percentiles_are_shares():
    peers = get_peer_stats("FPT", 2025)
    pcts = [m.value for m in peers.metrics if m.unit == "pctile"]
    assert pcts and all(0 < p <= 1 for p in pcts)


@needs_db
def test_forecast_removes_deconsolidated_subsidiary_from_the_base():
    f = get_forecast("FPT")
    reported, removed, base = (f.get(n, 2025).value for n in ("forecast_base_reported", "forecast_base_removed", "forecast_base"))
    assert base == pytest.approx(reported - removed)
    assert f.get("forecast_revenue_mid", 2026).value == pytest.approx(base * (1 + f.get("forecast_growth_mid", 2026).value))
    assert f.get("forecast_revenue_lo", 2026).value < f.get("forecast_revenue_mid", 2026).value < f.get("forecast_revenue_hi", 2026).value
    assert any("FPT Telecom" in n for n in f.notes)


@needs_db
def test_readers_see_true_ratios_while_the_model_keeps_its_clipped_inputs():
    from finresearch.db import query
    from finresearch.tools.risk import get_risk_signals

    shown = get_risk_signals("ELC", 2025).get("debt_g", 2025).value      # borrowings went from 36 to 396 billion
    model_input = query("SELECT value FROM metrics WHERE ticker = 'ELC' AND year = 2025 AND name = 'debt_g'").value.iloc[0]
    assert shown == pytest.approx(9.98, abs=0.05) and model_input == 5.0


@needs_db
def test_forecast_without_scope_change_uses_reported_revenue():
    f = get_forecast("HPG")
    assert f.get("forecast_base_reported", 2025) is None
    assert f.get("forecast_base", 2025).value == pytest.approx(get_financials("HPG", 2025, 2025).get("revenue", 2025).value)
