"""Turn extracted statement lines into one row per ticker-year with model features.

Every ratio and growth rate is computed inside a single report, from its
current-year and prior-year columns. That keeps a firm's numbers on one unit
and one restatement basis, so differences between reports cannot leak into a
feature. The target is next year's revenue growth, read the same way from the
following year's report.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .extract import fold

# name: (statement, candidate codes in order of preference, pattern the label must match)
FIELDS: dict[str, tuple[str, tuple[str, ...], str | None]] = {
    "revenue": ("IS", ("10",), r"doanh thu thuan|net revenue|net sales|net turnover"),
    "cogs": ("IS", ("11",), r"gia von|cost of"),
    "gross_profit": ("IS", ("20",), r"gop|gross"),
    "fin_income": ("IS", ("21",), r"tai chinh|financ"),
    "fin_expense": ("IS", ("22",), r"tai chinh|financ"),
    "interest_expense": ("IS", ("23",), r"lai vay|interest"),
    # code 24 is selling expense in separate statements and share of associates in consolidated ones
    "selling_exp": ("IS", ("25", "24"), r"ban hang|selling"),
    "admin_exp": ("IS", ("26", "25"), r"quan ly|administrat"),
    "op_profit": ("IS", ("30",), r"hoat dong kinh doanh|operating"),
    "pbt": ("IS", ("50",), r"truoc thue|before tax"),
    "net_income": ("IS", ("60",), r"sau thue|after tax|net profit|profit for the"),
    "current_assets": ("BS", ("100",), r"ngan han|current"),
    "cash": ("BS", ("110",), r"tien|cash"),
    "st_investments": ("BS", ("120",), r"dau tu|investment"),
    "st_receivables": ("BS", ("130",), r"phai thu|receivable"),
    "trade_receivables": ("BS", ("131",), r"phai thu|receivable"),
    "inventory": ("BS", ("140",), r"ton kho|inventor"),
    "noncurrent_assets": ("BS", ("200",), r"dai han|non-?current|long-?term"),
    "fixed_assets": ("BS", ("220",), r"co dinh|fixed assets"),
    "total_assets": ("BS", ("270",), r"tong (cong )?tai san|total assets"),
    "liabilities": ("BS", ("300",), r"no phai tra|liabilities"),
    "current_liabilities": ("BS", ("310",), r"ngan han|current"),
    "st_debt": ("BS", ("320",), r"vay|borrowing|loan"),
    "lt_debt": ("BS", ("338",), r"vay|borrowing|loan"),
    "equity": ("BS", ("400",), r"von chu so huu|equity|nguon von"),
    "funding_total": ("BS", ("440",), r"tong (cong )?nguon von|total"),
    "depreciation": ("CF", ("02",), r"khau hao|depreciation"),
    "cfo": ("CF", ("20",), r"kinh doanh|operating"),
    "capex": ("CF", ("21",), r"mua sam|xay dung|purchase|acquisition|construction"),
    "cfi": ("CF", ("30",), r"dau tu|investing"),
    "cff": ("CF", ("40",), r"tai chinh|financing"),
    "dividends_paid": ("CF", ("36",), r"co tuc|loi nhuan da tra|dividend"),
}

# Printed in brackets by some auditors and as plain amounts by others.
UNSIGNED = ("cogs", "fin_expense", "interest_expense", "selling_exp", "admin_exp", "capex", "dividends_paid", "depreciation")
# A blank line on a valid balance sheet means the firm has none of it.
ZERO_IF_MISSING = ("st_debt", "lt_debt", "inventory", "trade_receivables", "st_investments", "fixed_assets")

MIN_REVENUE = 1e10  # VND; below this, growth rates are noise
RESTATEMENT_TOL = np.log(1.25)


def build_wide(facts: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """One row per selected report with `<field>` and `<field>_prev` in VND."""
    f = facts[facts.file.isin(selected.file)].copy()
    wanted = {(st, c) for st, codes, _ in FIELDS.values() for c in codes}
    f = f[[(s, c) in wanted for s, c in zip(f.statement, f.code)]]
    f["flabel"] = f.label.map(fold)

    cols = {}
    for name, (st, codes, pattern) in FIELDS.items():
        cand = f[(f.statement == st) & f.code.isin(codes)]
        if pattern:
            cand = cand[cand.flabel.map(lambda t: bool(re.search(pattern, t)))]
        cand = cand.assign(_p=cand.code.map({c: i for i, c in enumerate(codes)})).sort_values("_p")
        cand = cand.drop_duplicates("file").set_index("file")
        cols[name] = cand.cur
        cols[name + "_prev"] = cand.prev
    wide = pd.DataFrame(cols).reindex(selected.file.values)
    wide.index = pd.MultiIndex.from_arrays([selected.ticker.values, selected.year.values], names=["ticker", "year"])

    meta = selected.set_index(["ticker", "year"])
    for name, (st, _, _) in FIELDS.items():
        unit = meta[f"unit_{st}"].fillna(1.0)
        ok = meta[f"{st.lower()}_ok"]
        for col in (name, name + "_prev"):
            v = wide[col] * unit
            if name in UNSIGNED:
                v = v.abs()
            if name in ZERO_IF_MISSING:
                v = v.fillna(0.0)
            wide[col] = v.where(ok)  # a statement that fails its identity is not trusted at all
    # recover total assets when only the funding total survived OCR
    for suffix in ("", "_prev"):
        parts = wide["current_assets" + suffix] + wide["noncurrent_assets" + suffix]
        wide["total_assets" + suffix] = (
            wide["total_assets" + suffix].fillna(wide["funding_total" + suffix]).fillna(parts)
        )
    wide = wide.drop(columns=["funding_total", "funding_total_prev"])
    return wide.join(meta[["file", "scope", "english", "bs_ok", "is_ok", "cf_ok"]]).sort_index()


def _div(a, b):
    """a / b where b > 0, else NaN."""
    b = b.where(b > 0)
    return a / b


def _growth(cur, prev):
    return _div(cur, prev) - 1.0


def add_features(wide: pd.DataFrame) -> pd.DataFrame:
    w = wide
    d = pd.DataFrame(index=w.index)
    avg_assets = (w.total_assets + w.total_assets_prev) / 2
    avg_equity = (w.equity + w.equity_prev) / 2
    debt = w.st_debt + w.lt_debt
    sga = w.selling_exp.fillna(0) + w.admin_exp.fillna(0)
    sga_prev = w.selling_exp_prev.fillna(0) + w.admin_exp_prev.fillna(0)

    # growth during year t
    d["rev_g"] = _growth(w.revenue, w.revenue_prev)
    d["assets_g"] = _growth(w.total_assets, w.total_assets_prev)
    d["equity_g"] = _growth(w.equity, w.equity_prev)
    d["liab_g"] = _growth(w.liabilities, w.liabilities_prev)
    d["recv_g"] = _growth(w.trade_receivables, w.trade_receivables_prev)
    d["inv_g"] = _growth(w.inventory, w.inventory_prev)
    d["fixed_g"] = _growth(w.fixed_assets, w.fixed_assets_prev)
    d["recv_minus_rev_g"] = d.recv_g - d.rev_g
    d["inv_minus_rev_g"] = d.inv_g - d.rev_g

    # profitability
    d["gross_margin"] = _div(w.gross_profit, w.revenue)
    d["op_margin"] = _div(w.op_profit, w.revenue)
    d["net_margin"] = _div(w.net_income, w.revenue)
    d["gross_margin_chg"] = d.gross_margin - _div(w.gross_profit_prev, w.revenue_prev)
    d["op_margin_chg"] = d.op_margin - _div(w.op_profit_prev, w.revenue_prev)
    d["net_margin_chg"] = d.net_margin - _div(w.net_income_prev, w.revenue_prev)
    d["sga_ratio"] = _div(sga, w.revenue)
    d["roe"] = _div(w.net_income, avg_equity)
    d["roa"] = _div(w.net_income, avg_assets)

    # cash flow quality and investment
    d["cfo_margin"] = _div(w.cfo, w.revenue)
    d["accruals"] = _div(w.net_income - w.cfo, avg_assets)
    d["capex_to_rev"] = _div(w.capex, w.revenue)
    d["capex_to_dep"] = _div(w.capex, w.depreciation)
    d["capex_to_fixed"] = _div(w.capex, w.fixed_assets_prev)
    d["fcf_margin"] = _div(w.cfo - w.capex, w.revenue)
    d["payout"] = _div(w.dividends_paid, w.net_income)
    d["cff_to_assets"] = _div(w.cff, avg_assets)

    # efficiency
    d["asset_turnover"] = _div(w.revenue, avg_assets)
    d["dso"] = 365 * _div(w.trade_receivables, w.revenue)
    d["dio"] = 365 * _div(w.inventory, w.cogs)
    d["dso_chg"] = d.dso - 365 * _div(w.trade_receivables_prev, w.revenue_prev)
    d["dio_chg"] = d.dio - 365 * _div(w.inventory_prev, w.cogs_prev)

    # balance sheet structure
    d["liab_to_assets"] = _div(w.liabilities, w.total_assets)
    d["debt_to_equity"] = _div(debt, w.equity)
    d["debt_g"] = _growth(debt, w.st_debt_prev + w.lt_debt_prev)
    d["current_ratio"] = _div(w.current_assets, w.current_liabilities)
    d["cash_to_assets"] = _div(w.cash + w.st_investments, w.total_assets)
    d["fixed_to_assets"] = _div(w.fixed_assets, w.total_assets)
    d["interest_cover"] = _div(w.pbt + w.interest_expense.fillna(0), w.interest_expense)
    d["log_revenue"] = np.log(w.revenue.where(w.revenue > 0))
    d["log_assets"] = np.log(w.total_assets.where(w.total_assets > 0))

    # Beneish M-Score (1999), eight-variable model
    ppe = w.fixed_assets
    d["dsri"] = _div(_div(w.trade_receivables, w.revenue), _div(w.trade_receivables_prev, w.revenue_prev))
    d["gmi"] = _div(_div(w.gross_profit_prev, w.revenue_prev), d.gross_margin)
    soft = 1 - _div(w.current_assets + ppe, w.total_assets)
    soft_prev = 1 - _div(w.current_assets_prev + w.fixed_assets_prev, w.total_assets_prev)
    d["aqi"] = _div(soft, soft_prev)
    d["sgi"] = _div(w.revenue, w.revenue_prev)
    # prior-year depreciation is the prior-year column of the cash flow statement
    dep_rate = _div(w.depreciation, w.depreciation + ppe)
    dep_rate_prev = _div(w.depreciation_prev, w.depreciation_prev + w.fixed_assets_prev)
    d["depi"] = _div(dep_rate_prev, dep_rate)
    d["sgai"] = _div(_div(sga, w.revenue), _div(sga_prev, w.revenue_prev))
    lev = _div(w.lt_debt + w.current_liabilities, w.total_assets)
    lev_prev = _div(w.lt_debt_prev + w.current_liabilities_prev, w.total_assets_prev)
    d["lvgi"] = _div(lev, lev_prev)
    d["tata"] = _div(w.net_income - w.cfo, w.total_assets)
    comp = d[["dsri", "gmi", "aqi", "sgi", "depi", "sgai", "lvgi"]].clip(0, 10).fillna(1.0)  # 1.0 = no change
    d["m_score"] = (
        -4.84 + 0.92 * comp.dsri + 0.528 * comp.gmi + 0.404 * comp.aqi + 0.892 * comp.sgi
        + 0.115 * comp.depi - 0.172 * comp.sgai + 4.679 * d.tata.clip(-1, 1) - 0.327 * comp.lvgi
    ).where(d.tata.notna() & d.sgi.notna())

    return d.replace([np.inf, -np.inf], np.nan)


# Values outside these ranges are data errors or shell companies, not information.
# First matching pattern wins.
CLIPS = [
    (r"^log_|^m_score$", None),
    (r"^(dso|dio)$", (0, 1500)),
    (r"^(dso|dio)_chg$", (-1500, 1500)),
    (r"_chg$", (-1.0, 1.0)),
    (r"margin", (-2.0, 1.0)),
    (r"_g($|_)", (-0.95, 5.0)),
    (r"^(roe|roa|accruals|tata|cff_to_assets)$", (-2.0, 2.0)),
    (r"^interest_cover$", (-50, 200)),
    (r".", (0, 50)),  # remaining ratios are non-negative by construction
]


def clip_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    for col in d.columns:
        bounds = next(b for pat, b in CLIPS if re.search(pat, col))
        if bounds is not None:
            d[col] = d[col].clip(*bounds)
    return d


def build_panel(wide: pd.DataFrame, companies: pd.DataFrame | None = None) -> pd.DataFrame:
    """Model table from `build_wide` output: features for year t, target = log revenue growth from t to t+1."""
    feats = clip_features(add_features(wide))
    p = feats.join(wide[["revenue", "revenue_prev", "total_assets", "net_income", "cfo", "scope", "is_ok", "bs_ok", "cf_ok"]])
    p = p.reset_index()

    # values from the neighbouring years' reports
    nxt = p[["ticker", "year", "revenue", "revenue_prev", "scope", "is_ok"]].copy()
    nxt["year"] -= 1
    nxt.columns = ["ticker", "year", "revenue_next", "revenue_next_base", "scope_next", "is_ok_next"]
    p = p.merge(nxt, on=["ticker", "year"], how="left")
    for lag in (1, 2):
        prev = p[["ticker", "year", "rev_g", "gross_margin"]].copy()
        prev["year"] += lag
        prev.columns = ["ticker", "year", f"rev_g_lag{lag}", f"gross_margin_lag{lag}"]
        p = p.merge(prev, on=["ticker", "year"], how="left")
    p["rev_g_3y_mean"] = p[["rev_g", "rev_g_lag1", "rev_g_lag2"]].mean(axis=1)
    p["rev_g_3y_std"] = p[["rev_g", "rev_g_lag1", "rev_g_lag2"]].std(axis=1)

    growth_ok = (p.revenue_next > 0) & (p.revenue_next_base > 0)
    p["target"] = np.log((p.revenue_next / p.revenue_next_base).where(growth_ok))
    # next year's report restates this year's revenue; a large gap means the two
    # reports do not describe the same entity (or one of them was misread)
    gap = np.log((p.revenue_next_base / p.revenue).where((p.revenue > 0) & (p.revenue_next_base > 0))).abs()
    p["restated"] = gap > RESTATEMENT_TOL
    p["usable"] = p.is_ok & (p.revenue >= MIN_REVENUE) & (p.revenue_prev > 0)
    p["has_target"] = p.usable & p.target.notna() & ~p.restated & p.is_ok_next.eq(True) & (p.scope_next == p.scope)

    if companies is None:  # no reference table: every firm falls in one sector
        companies = pd.DataFrame(columns=["ticker", "exchange", "com_type", "icb1", "icb2"])
    p = p.merge(companies[["ticker", "exchange", "com_type", "icb1", "icb2"]], on="ticker", how="left")
    p["sector"] = p.icb2.fillna("Không rõ")
    stats = p[p.usable].groupby(["sector", "year"]).rev_g.agg(sector_rev_g="median", sector_n="size").reset_index()
    p = p.merge(stats, on=["sector", "year"], how="left")
    p = p.merge(p[p.usable].groupby("year").rev_g.median().rename("market_rev_g").reset_index(), on="year", how="left")
    return p.drop(columns=["revenue_next", "revenue_next_base", "scope_next", "is_ok_next"])


NON_FEATURES = {
    "ticker", "year", "target", "usable", "has_target", "restated", "scope", "is_ok", "bs_ok", "cf_ok",
    "revenue", "revenue_prev", "total_assets", "net_income", "cfo", "exchange", "com_type", "icb1", "icb2",
    "sector", "sector_n", "m_score", "sgi", "tata",
    "market_rev_g",  # one value per year: it would only let the model memorise years
}


def feature_columns(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c not in NON_FEATURES]


# Display names for reports and SHAP plots.
FEATURE_LABELS = {
    "rev_g": "Tăng trưởng doanh thu năm nay",
    "assets_g": "Tăng trưởng tổng tài sản",
    "equity_g": "Tăng trưởng vốn chủ sở hữu",
    "liab_g": "Tăng trưởng nợ phải trả",
    "recv_g": "Tăng trưởng phải thu khách hàng",
    "inv_g": "Tăng trưởng hàng tồn kho",
    "fixed_g": "Tăng trưởng tài sản cố định",
    "recv_minus_rev_g": "Phải thu tăng nhanh hơn doanh thu",
    "inv_minus_rev_g": "Tồn kho tăng nhanh hơn doanh thu",
    "gross_margin": "Biên lợi nhuận gộp",
    "op_margin": "Biên lợi nhuận hoạt động",
    "net_margin": "Biên lợi nhuận ròng",
    "gross_margin_chg": "Thay đổi biên gộp",
    "op_margin_chg": "Thay đổi biên hoạt động",
    "net_margin_chg": "Thay đổi biên ròng",
    "sga_ratio": "Chi phí bán hàng và quản lý / doanh thu",
    "roe": "ROE",
    "roa": "ROA",
    "cfo_margin": "Dòng tiền kinh doanh / doanh thu",
    "accruals": "Dồn tích (LNST − dòng tiền KD) / tài sản",
    "capex_to_rev": "Capex / doanh thu",
    "capex_to_dep": "Capex / khấu hao",
    "capex_to_fixed": "Capex / tài sản cố định đầu năm",
    "fcf_margin": "Dòng tiền tự do / doanh thu",
    "payout": "Cổ tức đã trả / LNST",
    "cff_to_assets": "Dòng tiền tài chính / tài sản",
    "asset_turnover": "Vòng quay tài sản",
    "dso": "Số ngày phải thu",
    "dio": "Số ngày tồn kho",
    "dso_chg": "Thay đổi số ngày phải thu",
    "dio_chg": "Thay đổi số ngày tồn kho",
    "liab_to_assets": "Nợ phải trả / tài sản",
    "debt_to_equity": "Vay / vốn chủ sở hữu",
    "debt_g": "Tăng trưởng vay",
    "current_ratio": "Hệ số thanh toán hiện hành",
    "cash_to_assets": "Tiền và đầu tư ngắn hạn / tài sản",
    "fixed_to_assets": "Tài sản cố định / tài sản",
    "interest_cover": "Khả năng trả lãi",
    "log_revenue": "Quy mô doanh thu (log)",
    "log_assets": "Quy mô tài sản (log)",
    "dsri": "Beneish: chỉ số phải thu",
    "gmi": "Beneish: chỉ số biên gộp",
    "aqi": "Beneish: chất lượng tài sản",
    "depi": "Beneish: chỉ số khấu hao",
    "sgai": "Beneish: chỉ số chi phí bán hàng và quản lý",
    "lvgi": "Beneish: chỉ số đòn bẩy",
    "rev_g_lag1": "Tăng trưởng doanh thu năm trước",
    "gross_margin_lag1": "Biên gộp năm trước",
    "rev_g_lag2": "Tăng trưởng doanh thu 2 năm trước",
    "gross_margin_lag2": "Biên gộp 2 năm trước",
    "rev_g_3y_mean": "Tăng trưởng doanh thu bình quân 3 năm",
    "rev_g_3y_std": "Độ biến động tăng trưởng 3 năm",
    "sector_rev_g": "Tăng trưởng trung vị của ngành năm nay",
}

FEATURE_LABELS_EN = {
    "rev_g": "Revenue growth this year",
    "assets_g": "Total asset growth",
    "equity_g": "Equity growth",
    "liab_g": "Growth in liabilities",
    "recv_g": "Growth in trade receivables",
    "inv_g": "Inventory growth",
    "fixed_g": "Growth in fixed assets",
    "recv_minus_rev_g": "Receivables growing faster than revenue",
    "inv_minus_rev_g": "Inventory growing faster than revenue",
    "gross_margin": "Gross margin",
    "op_margin": "Operating margin",
    "net_margin": "Net margin",
    "gross_margin_chg": "Change in gross margin",
    "op_margin_chg": "Change in operating margin",
    "net_margin_chg": "Change in net margin",
    "sga_ratio": "Selling and admin expenses / revenue",
    "roe": "ROE",
    "roa": "ROA",
    "cfo_margin": "Operating cash flow / revenue",
    "accruals": "Accruals (net profit − operating cash flow) / assets",
    "capex_to_rev": "Capex / revenue",
    "capex_to_dep": "Capex / depreciation",
    "capex_to_fixed": "Capex / opening fixed assets",
    "fcf_margin": "Free cash flow / revenue",
    "payout": "Dividends paid / net profit",
    "cff_to_assets": "Financing cash flow / assets",
    "asset_turnover": "Asset turnover",
    "dso": "Days sales outstanding",
    "dio": "Days inventory outstanding",
    "dso_chg": "Change in days sales outstanding",
    "dio_chg": "Change in days inventory outstanding",
    "liab_to_assets": "Liabilities / assets",
    "debt_to_equity": "Debt / equity",
    "debt_g": "Growth in borrowings",
    "current_ratio": "Current ratio",
    "cash_to_assets": "Cash and short-term investments / assets",
    "fixed_to_assets": "Fixed assets / assets",
    "interest_cover": "Interest cover",
    "log_revenue": "Revenue size (log)",
    "log_assets": "Asset size (log)",
    "dsri": "Beneish: receivables index",
    "gmi": "Beneish: gross margin index",
    "aqi": "Beneish: asset quality index",
    "depi": "Beneish: depreciation index",
    "sgai": "Beneish: selling and admin expense index",
    "lvgi": "Beneish: leverage index",
    "rev_g_lag1": "Revenue growth last year",
    "gross_margin_lag1": "Gross margin last year",
    "rev_g_lag2": "Revenue growth two years ago",
    "gross_margin_lag2": "Gross margin two years ago",
    "rev_g_3y_mean": "Average revenue growth over three years",
    "rev_g_3y_std": "Volatility of revenue growth over three years",
    "sector_rev_g": "Median revenue growth of the sector this year",
}
