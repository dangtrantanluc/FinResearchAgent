# %% [markdown]
# # FinResearch Agent — mô hình dự báo tăng trưởng doanh thu
#
# Notebook này làm trọn phần ML của project:
#
# 1. Tải bộ `vduydong/ocr_annual_financials` (BCTC kiểm toán 2015–2025, bản OCR) và trích số liệu theo mã số Thông tư 200.
# 2. Dựng bảng đặc trưng: mỗi dòng là một công ty trong một năm.
# 3. Train XGBoost dự báo **tăng trưởng doanh thu năm sau so với mặt bằng chung**, so với baseline, đánh giá theo thời gian.
# 4. Khoảng dự báo bằng conformal prediction.
# 5. Giải thích bằng SHAP và đo độ ổn định của lời giải thích qua bootstrap.
# 6. Dự báo 2026 và lưu artifact để app dùng lại.
#
# **Chạy trên Kaggle:** bật *Internet* trong Settings rồi Run All. Không cần GPU. Lần chạy đầu mất khoảng 10–15 phút, phần lớn là tải và trích dữ liệu.

# %%
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)

try:
    display
except NameError:  # running as a plain script
    display = print

ON_KAGGLE = Path("/kaggle/working").exists()
ROOT = next((p for p in (Path.cwd(), Path.cwd().parent) if (p / "finresearch").exists()), Path.cwd())
sys.path.insert(0, str(ROOT))

RAW_DIR = Path(os.environ.get("FINRESEARCH_RAW", "/tmp/ocr_annual_financials" if ON_KAGGLE else ROOT / "data/raw/ocr_annual_financials"))
DATA_DIR = Path("/kaggle/working/processed") if ON_KAGGLE else ROOT / "data/processed"
OUT_DIR = Path("/kaggle/working/artifacts") if ON_KAGGLE else ROOT / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REBUILD = False  # True: trích lại từ đầu dù đã có panel.parquet
SEED = 42
FOCUS = "FPT"  # công ty dùng làm ví dụ giải thích

from finresearch.build import build_dataset, download_raw
from finresearch.panel import FEATURE_LABELS, feature_columns

print("xgboost", xgb.__version__, "| pandas", pd.__version__, "| raw:", RAW_DIR)

# %% [markdown]
# ## 1. Dữ liệu
#
# Bộ dữ liệu gồm 18.231 file text, mỗi file là một báo cáo. Các báo cáo tài chính nằm trong bảng HTML có cột **Mã số** (ví dụ 10 = doanh thu thuần, 270 = tổng tài sản) cùng hai cột số: năm nay và năm trước.
#
# Bước trích xuất (`finresearch/extract.py`) tự tìm cột mã số, phân loại bảng thành cân đối kế toán / kết quả kinh doanh / lưu chuyển tiền tệ, rồi kiểm tra các đẳng thức kế toán:
#
# - Tổng tài sản = Tổng nguồn vốn
# - Lợi nhuận gộp = Doanh thu thuần − Giá vốn
# - Lưu chuyển tiền thuần = Kinh doanh + Đầu tư + Tài chính
#
# Báo cáo nào không khớp trong sai số 0,5% thì phần đó bị loại. Ngân hàng, chứng khoán, bảo hiểm dùng mẫu biểu khác nên cũng bị loại ở bước kiểm tra nhãn.

# %%
panel_path = DATA_DIR / "panel.parquet"
if REBUILD or not panel_path.exists():
    download_raw(RAW_DIR)
    companies_csv = ROOT / "reference/companies.csv"
    graded, selected, panel = build_dataset(RAW_DIR, DATA_DIR, companies_csv, workers=os.cpu_count() or 2)
else:
    panel = pd.read_parquet(panel_path)
    graded = pd.read_parquet(DATA_DIR / "reports.parquet")

corp = graded[graded.corporate]
quality = pd.Series({
    "Số file": f"{len(graded):,}",
    "Đúng mẫu Thông tư 200": f"{len(corp):,}",
    "Cân đối kế toán khớp": f"{corp.bs_ok.mean():.1%}",
    "Kết quả kinh doanh khớp": f"{corp.is_ok.mean():.1%}",
    "Lưu chuyển tiền tệ khớp": f"{corp.cf_ok.mean():.1%}",
    "Dòng công ty–năm": f"{len(panel):,}",
    "Dòng có nhãn": f"{int(panel.has_target.sum()):,}",
    "Số mã có nhãn": f"{panel[panel.has_target].ticker.nunique():,}",
})
display(quality.to_frame("giá trị"))

# %% [markdown]
# Kiểm tra nhanh với một công ty quen thuộc. Doanh thu thuần của FPT theo BCTC kiểm toán là 44.010 tỷ (2022), 52.618 tỷ (2023) và 62.849 tỷ (2024).

# %%
focus = panel[panel.ticker == FOCUS].set_index("year")
check = (focus[["revenue", "net_income", "total_assets", "cfo"]] / 1e9).round(0)
check.columns = ["Doanh thu thuần", "LNST", "Tổng tài sản", "Dòng tiền KD"]
display(check.T.apply(lambda s: s.map("{:,.0f}".format)))

# %% [markdown]
# ## 2. Bài toán
#
# **Đầu vào:** các chỉ số của công ty trong năm *t*, tính từ chính báo cáo năm *t* (cột năm nay và cột năm trước).
#
# **Nhãn:** log tăng trưởng doanh thu từ *t* sang *t+1*, lấy từ báo cáo năm *t+1*, **trừ đi trung vị của tất cả công ty trong cùng năm**.
#
# Vì sao trừ trung vị năm: mặt bằng tăng trưởng mỗi năm do vĩ mô quyết định (2021 cao, 2022 thấp) và không thể đoán được từ BCTC của từng công ty. Phần mô hình học được là *công ty nào tăng nhanh hơn hay chậm hơn số đông*. Khi cần con số tuyệt đối thì cộng thêm một giả định về mặt bằng chung.
#
# Mọi tỷ lệ đều tính trong một báo cáo, nên đơn vị (đồng, nghìn đồng, triệu đồng) tự triệt tiêu và việc điều chỉnh hồi tố giữa các năm không làm lệch đặc trưng.

# %%
FEATURES = feature_columns(panel)
CLIP = 1.0  # log tăng trưởng bị cắt ở ±1 (khoảng −63% … +172%) để M&A và thoái vốn không chi phối

data = panel[panel.has_target].copy()
data["y_abs"] = data.target.clip(-CLIP, CLIP)
data["year_median"] = data.groupby("year").y_abs.transform("median")
data["y"] = data.y_abs - data.year_median

DEV_END, TEST_YEARS = 2022, [2023, 2024]
dev = data[data.year <= DEV_END]
test = data[data.year.isin(TEST_YEARS)]
print(f"{len(FEATURES)} đặc trưng | dev (t ≤ {DEV_END}): {len(dev):,} dòng | test (t = {TEST_YEARS}): {len(test):,} dòng")
display(data.groupby("year").agg(so_dong=("y", "size"), trung_vi_tang_truong=("y_abs", "median"), do_lech_chuan=("y", "std")).T)

fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
data.y_abs.hist(bins=80, ax=ax[0], color="#4C72B0")
ax[0].set_title("Log tăng trưởng doanh thu năm sau")
ax[0].set_xlabel("log(DT năm t+1 / DT năm t)")
data.groupby("year").y_abs.median().plot(marker="o", ax=ax[1], color="#C44E52")
ax[1].axhline(0, color="grey", lw=0.8)
ax[1].set_title("Trung vị theo năm — phần mô hình không dự báo")
ax[1].set_xlabel("năm t")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Chọn cấu hình bằng cross-validation theo thời gian
#
# Dữ liệu là chuỗi theo năm nên không xáo trộn ngẫu nhiên. Mỗi fold train trên các năm trước *T* và kiểm tra trên năm *T*, với *T* = 2019 … 2022. Số vòng boosting được chọn tại điểm sai số trung bình của bốn fold thấp nhất.

# %%
CV_YEARS = [2019, 2020, 2021, 2022]
BASE = dict(learning_rate=0.03, subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0,
            tree_method="hist", objective="reg:squarederror", random_state=SEED, n_jobs=os.cpu_count())
GRID = [dict(max_depth=d, min_child_weight=w) for d in (3, 4, 6) for w in (10, 30)]
MAX_ROUNDS = 800


def forward_cv(df, params, years=CV_YEARS, rounds=MAX_ROUNDS):
    """MAE trung bình theo số vòng boosting, qua các fold tiến theo thời gian."""
    curves = []
    for T in years:
        tr, va = df[df.year < T], df[df.year == T]
        m = xgb.XGBRegressor(n_estimators=rounds, eval_metric="mae", **BASE, **params)
        m.fit(tr[FEATURES], tr.y, eval_set=[(va[FEATURES], va.y)], verbose=False)
        curves.append(m.evals_result()["validation_0"]["mae"])
    return np.mean(curves, axis=0)


rows = []
for params in GRID:
    curve = forward_cv(dev, params)
    rows.append({**params, "rounds": int(curve.argmin()) + 1, "cv_mae": curve.min()})
cv_table = pd.DataFrame(rows).sort_values("cv_mae").reset_index(drop=True)
display(cv_table)

best = cv_table.iloc[0]
PARAMS = dict(max_depth=int(best.max_depth), min_child_weight=int(best.min_child_weight))
ROUNDS = int(best.rounds)
print("Chọn:", PARAMS, "| số vòng:", ROUNDS)


def fit_model(df, seed=SEED):
    return xgb.XGBRegressor(n_estimators=ROUNDS, **{**BASE, "random_state": seed}, **PARAMS).fit(df[FEATURES], df.y)


# %% [markdown]
# ## 4. Kết quả trên tập test
#
# Tập test là hai năm mô hình chưa từng thấy: 2023→2024 và 2024→2025. So với bốn baseline:
#
# - **Bằng mặt bằng chung:** mọi công ty tăng đúng bằng trung vị năm (dự báo 0).
# - **Giữ nguyên đà:** năm sau lệch khỏi mặt bằng chung đúng như năm nay.
# - **Theo ngành:** năm sau lệch đúng bằng mức ngành đang lệch.
# - **Ridge:** hồi quy tuyến tính trên cùng bộ đặc trưng.
#
# **IC** là tương quan hạng Spearman giữa dự báo và thực tế trong từng năm: đo khả năng xếp đúng thứ tự công ty tăng nhanh / chậm.

# %%
model = fit_model(dev)
ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 20)))
ridge.fit(dev[FEATURES], dev.y)


def rel_now(df):
    """Tăng trưởng năm nay so với trung vị năm nay, cùng thang log với nhãn."""
    return (np.log1p(df.rev_g.clip(-0.6, 1.7)) - np.log1p(df.market_rev_g)).fillna(0)


def predictions(df):
    return {
        "Bằng mặt bằng chung": np.zeros(len(df)),
        "Giữ nguyên đà": rel_now(df).values,
        "Theo ngành": (np.log1p(df.sector_rev_g.fillna(df.market_rev_g)) - np.log1p(df.market_rev_g)).values,
        "Ridge": ridge.predict(df[FEATURES]),
        "XGBoost": model.predict(df[FEATURES]),
    }


def score(df, preds):
    rows = []
    zero_mae = np.abs(df.y).mean()
    for name, p in preds.items():
        p = np.clip(p, -CLIP, CLIP)
        ics = [spearmanr(p[(df.year == yr).values], df.y[df.year == yr]).statistic for yr in sorted(df.year.unique())] if np.std(p) > 0 else [np.nan]
        mae = np.abs(df.y - p).mean()
        rows.append({"mô hình": name, "MAE": mae, "cải thiện so với mặt bằng chung": 1 - mae / zero_mae,
                     "RMSE": np.sqrt(((df.y - p) ** 2).mean()), "IC trung bình": np.mean(ics)})
    return pd.DataFrame(rows).set_index("mô hình")


test_preds = predictions(test)
test_scores = score(test, test_preds)
display(test_scores)

# %% [markdown]
# Cách đọc dễ hơn: chia công ty trong tập test thành 5 nhóm theo dự báo, rồi xem tăng trưởng thực tế của từng nhóm. Nếu mô hình có ích thì nhóm được dự báo cao nhất phải tăng thật sự nhanh hơn nhóm thấp nhất.

# %%
q = test.assign(pred=test_preds["XGBoost"])
q["nhom"] = q.groupby("year").pred.transform(lambda s: pd.qcut(s, 5, labels=False, duplicates="drop")) + 1
by_q = q.groupby("nhom").agg(du_bao=("pred", "median"), thuc_te=("y", "median"), so_cong_ty=("y", "size"))
display(by_q)

fig, ax = plt.subplots(figsize=(6, 3.4))
(np.expm1(by_q.thuc_te) * 100).plot.bar(ax=ax, color="#4C72B0")
ax.axhline(0, color="grey", lw=0.8)
ax.set_xlabel("nhóm theo dự báo (1 = thấp nhất, 5 = cao nhất)")
ax.set_ylabel("tăng trưởng thực tế so với trung vị (%)")
ax.set_title("Tập test: tăng trưởng thực tế theo nhóm dự báo")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Backtest tiến dần
#
# Lặp lại cho từng năm: train trên mọi năm trước *T*, dự báo năm *T*. Một năm test tốt có thể là may; sáu năm thì khó may hơn.
#
# Lưu ý: các năm 2019–2022 cũng là các fold đã dùng để chọn cấu hình ở mục 3, nên chỉ 2023 và 2024 là hoàn toàn ngoài mẫu.

# %%
rows, oof = [], []
for T in range(2019, 2025):
    tr, te = data[data.year < T], data[data.year == T]
    m = fit_model(tr)
    p = m.predict(te[FEATURES])
    oof.append(te[["ticker", "year", "y", "rev_g_3y_std"]].assign(pred=p))
    rows.append({"năm t": T, "số công ty": len(te), "MAE mặt bằng chung": np.abs(te.y).mean(), "MAE XGBoost": np.abs(te.y - p).mean(),
                 "IC": spearmanr(p, te.y).statistic})
walk = pd.DataFrame(rows).set_index("năm t")
walk["cải thiện"] = 1 - walk["MAE XGBoost"] / walk["MAE mặt bằng chung"]
display(walk)
oof = pd.concat(oof)

# %% [markdown]
# ## 5. Khoảng dự báo (conformal)
#
# Một con số dự báo mà không kèm độ chắc chắn thì dễ gây hiểu lầm. Cách làm: lấy sai số tuyệt đối của các dự báo ngoài mẫu trong giai đoạn dev (backtest 2019–2022), rồi dùng phân vị 80% của chúng làm bán kính khoảng.
#
# Một bán kính chung cho mọi công ty thì quá thô: công ty có doanh thu ổn định dễ dự báo hơn hẳn công ty lên xuống thất thường. Vì vậy bán kính được tính riêng cho từng nhóm theo **độ biến động tăng trưởng 3 năm gần nhất** của chính công ty đó (Mondrian conformal). Sau đó kiểm tra trên tập test xem mỗi nhóm có thật sự phủ khoảng 80% giá trị thực không.

# %%
COVERAGE = 0.80
VOL_BINS = [-np.inf, 0.05, 0.10, 0.20, 0.40, np.inf]
VOL_LABELS = ["≤ 5%", "5–10%", "10–20%", "20–40%", "> 40%"]
NO_HISTORY = "thiếu lịch sử"


def vol_group(df):
    """Nhóm theo độ lệch chuẩn tăng trưởng doanh thu 3 năm; công ty mới thì vào nhóm riêng."""
    return pd.cut(df.rev_g_3y_std, VOL_BINS, labels=VOL_LABELS).astype(object).fillna(NO_HISTORY)


def conformal_radius(abs_resid, coverage=COVERAGE):
    r = np.sort(np.asarray(abs_resid))
    k = int(np.ceil((len(r) + 1) * coverage)) - 1
    return float(r[min(k, len(r) - 1)])


oof["group"] = vol_group(oof)
oof["abs_resid"] = (oof.y - oof.pred).abs()
calib = oof[oof.year <= DEV_END]
RADIUS = calib.groupby("group").abs_resid.apply(conformal_radius)
RADIUS_GLOBAL = conformal_radius(calib.abs_resid)

t = test.assign(pred=test_preds["XGBoost"])
t["group"] = vol_group(t)
t["covered"] = (t.y - t.pred).abs() <= t.group.map(RADIUS)
order = VOL_LABELS + [NO_HISTORY]
interval_table = pd.DataFrame({
    "số dòng hiệu chỉnh": calib.groupby("group").size(),
    "bán kính (log)": RADIUS,
    "cận dưới": np.expm1(-RADIUS),
    "cận trên": np.expm1(RADIUS),
    "số dòng test": t.groupby("group").size(),
    "tỷ lệ phủ trên test": t.groupby("group").covered.mean(),
}).reindex(order)
display(interval_table)
covered = float(t.covered.mean())
covered_global = float(((t.y - t.pred).abs() <= RADIUS_GLOBAL).mean())
print(f"Tỷ lệ phủ trên toàn tập test: {covered:.1%} (mục tiêu {COVERAGE:.0%})")
print(f"Nếu dùng một bán kính chung ±{RADIUS_GLOBAL:.3f}: phủ {covered_global:.1%}, nhưng quá rộng với công ty ổn định và quá hẹp với công ty biến động")

# %% [markdown]
# ## 6. Giải thích mô hình bằng SHAP
#
# Giá trị SHAP cho biết mỗi đặc trưng đẩy dự báo của một công ty lên hay xuống bao nhiêu so với mức trung bình. Ở đây dùng TreeSHAP có sẵn trong XGBoost (`pred_contribs`), cho kết quả chính xác với mô hình cây.

# %%
try:
    import shap
except ImportError:
    os.system(f"{sys.executable} -m pip install -q shap")
    import shap


def shap_values(m, X):
    """Trả về (giá trị SHAP, giá trị nền). Tổng mỗi dòng + nền = dự báo."""
    contrib = m.get_booster().predict(xgb.DMatrix(X[FEATURES]), pred_contribs=True)
    return contrib[:, :-1], float(contrib[0, -1])


LABELS = [FEATURE_LABELS.get(f, f) for f in FEATURES]
sv, base_value = shap_values(model, test)
explanation = shap.Explanation(values=sv, base_values=np.full(len(test), base_value), data=test[FEATURES].values, feature_names=LABELS)
global_imp = pd.Series(np.abs(sv).mean(axis=0), index=FEATURES).sort_values(ascending=False)

shap.plots.beeswarm(explanation, max_display=15, show=False)
plt.title("Ảnh hưởng của từng đặc trưng (tập test)")
plt.xlabel("Giá trị SHAP (log tăng trưởng so với mặt bằng chung)")
plt.tight_layout()
plt.show()
display(global_imp.head(15).rename(FEATURE_LABELS).to_frame("trung bình |SHAP|"))

# %% [markdown]
# ### Giải thích cho một công ty
#
# Dự báo cho năm 2026 dùng đặc trưng của năm 2025. Mô hình cuối được train lại trên toàn bộ dữ liệu có nhãn (đến 2024→2025).

# %%
final_model = fit_model(data)
latest_year = int(panel.year.max())
latest = panel[(panel.year == latest_year) & panel.usable].copy()
latest["pred_rel"] = final_model.predict(latest[FEATURES])

# mặt bằng chung là giả định, không phải dự báo: lấy trung vị của các trung vị năm trong quá khứ
MARKET_ASSUMPTION = float(data.groupby("year").y_abs.median().median())
latest["vol_group"] = vol_group(latest)
latest["radius"] = latest.vol_group.map(RADIUS).astype(float)
latest["growth_mid"] = np.expm1(MARKET_ASSUMPTION + latest.pred_rel)
latest["growth_lo"] = np.expm1(MARKET_ASSUMPTION + latest.pred_rel - latest.radius)
latest["growth_hi"] = np.expm1(MARKET_ASSUMPTION + latest.pred_rel + latest.radius)
latest["revenue_next_mid"] = latest.revenue * (1 + latest.growth_mid)

row = latest[latest.ticker == FOCUS]
if len(row):
    r = row.iloc[0]
    print(f"{FOCUS}: doanh thu {latest_year} = {r.revenue / 1e9:,.0f} tỷ")
    print(f"  Giả định mặt bằng chung {latest_year + 1}: {np.expm1(MARKET_ASSUMPTION):+.1%}")
    print(f"  Mô hình: {FOCUS} lệch {np.expm1(r.pred_rel):+.1%} so với mặt bằng chung (nhóm biến động: {r.vol_group})")
    print(f"  Tăng trưởng {latest_year + 1}: {r.growth_mid:+.1%} (khoảng {COVERAGE:.0%}: {r.growth_lo:+.1%} đến {r.growth_hi:+.1%})")
    print(f"  Doanh thu {latest_year + 1}: {r.revenue_next_mid / 1e9:,.0f} tỷ (khoảng: {r.revenue * (1 + r.growth_lo) / 1e9:,.0f} – {r.revenue * (1 + r.growth_hi) / 1e9:,.0f} tỷ)")

    sv_f, base_f = shap_values(final_model, row)
    # nhân 100 để đọc theo điểm phần trăm; ở thang log gốc các giá trị quá nhỏ và bị làm tròn về 0
    shap.plots.waterfall(shap.Explanation(values=sv_f[0] * 100, base_values=base_f * 100, data=row[FEATURES].values[0], feature_names=LABELS), max_display=12, show=False)
    plt.title(f"{FOCUS}: đóng góp của từng đặc trưng vào dự báo {latest_year + 1} (điểm %, so với mặt bằng chung)")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Lời giải thích có ổn định không?
#
# Nếu train lại trên một mẫu dữ liệu hơi khác mà danh sách "yếu tố quan trọng nhất" đổi hẳn thì lời giải thích đó không đáng tin. Ở đây train lại 30 lần, mỗi lần lấy mẫu lại **theo công ty** (bootstrap theo cụm), rồi đo:
#
# - Thứ hạng độ quan trọng toàn cục có giống nhau giữa các lần không (tương quan hạng Spearman).
# - Top 10 đặc trưng trùng nhau bao nhiêu.
# - Với công ty ví dụ: giá trị SHAP của từng đặc trưng dao động bao nhiêu, và có giữ nguyên dấu không.

# %%
N_BOOT = 30
rng = np.random.default_rng(SEED)
tickers = data.ticker.unique()
by_ticker = {t: g for t, g in data.groupby("ticker")}
boot_imp, boot_local, boot_pred = [], [], []
for b in range(N_BOOT):
    sample = pd.concat([by_ticker[t] for t in rng.choice(tickers, len(tickers), replace=True)])
    m = fit_model(sample, seed=SEED + 1 + b)
    s, _ = shap_values(m, test)
    boot_imp.append(np.abs(s).mean(axis=0))
    if len(row):
        s_f, _ = shap_values(m, row)
        boot_local.append(s_f[0])
        boot_pred.append(float(m.predict(row[FEATURES])[0]))
boot_imp = pd.DataFrame(boot_imp, columns=FEATURES)

rank_corr = boot_imp.T.corr(method="spearman").values
pairwise = rank_corr[np.triu_indices(N_BOOT, k=1)]
top10 = set(global_imp.index[:10])
overlap = [len(top10 & set(boot_imp.iloc[b].sort_values(ascending=False).index[:10])) / 10 for b in range(N_BOOT)]
stability = {"rank_corr_mean": float(pairwise.mean()), "rank_corr_min": float(pairwise.min()), "top10_overlap_mean": float(np.mean(overlap))}
print(f"Tương quan hạng giữa các lần train: trung bình {stability['rank_corr_mean']:.2f}, thấp nhất {stability['rank_corr_min']:.2f}")
print(f"Top 10 trùng với mô hình chính: trung bình {stability['top10_overlap_mean']:.0%}")

imp_summary = pd.DataFrame({"mô hình chính": global_imp, "bootstrap trung bình": boot_imp.mean(), "bootstrap độ lệch chuẩn": boot_imp.std()})
imp_summary["hệ số biến thiên"] = imp_summary["bootstrap độ lệch chuẩn"] / imp_summary["bootstrap trung bình"]
display(imp_summary.sort_values("mô hình chính", ascending=False).head(12).rename(FEATURE_LABELS))

local_summary = None
if len(row):
    boot_local = pd.DataFrame(boot_local, columns=FEATURES)
    local_summary = pd.DataFrame({
        "giá trị": row[FEATURES].iloc[0],
        "SHAP mô hình chính": sv_f[0],
        "SHAP bootstrap trung bình": boot_local.mean(),
        "độ lệch chuẩn": boot_local.std(),
        "cùng dấu với mô hình chính": (np.sign(boot_local) == np.sign(sv_f[0])).mean(),
    }, index=FEATURES)
    local_summary = local_summary.reindex(local_summary["SHAP mô hình chính"].abs().sort_values(ascending=False).index)
    print(f"\n{FOCUS}: dự báo lệch so với mặt bằng chung qua {N_BOOT} lần train: {np.mean(boot_pred):+.3f} ± {np.std(boot_pred):.3f} log")
    display(local_summary.head(10).rename(FEATURE_LABELS))

    top = local_summary.head(10).iloc[::-1].rename(FEATURE_LABELS)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(top.index, top["SHAP bootstrap trung bình"], xerr=top["độ lệch chuẩn"], color="#4C72B0", ecolor="#333", capsize=3)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_title(f"{FOCUS}: SHAP trung bình ± độ lệch chuẩn qua {N_BOOT} lần bootstrap")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 7. Tín hiệu rủi ro kế toán: Beneish M-Score
#
# M-Score (Beneish, 1999) gộp tám tỷ số thành một điểm; trên −1,78 thường được coi là dấu hiệu nên xem kỹ chất lượng lợi nhuận. Đây là công thức cố định, không train. Ngưỡng được ước lượng trên doanh nghiệp Mỹ nên ở đây chỉ dùng để **xếp hạng trong cùng ngành, cùng năm**, không dùng như kết luận.

# %%
risk = panel[panel.usable & panel.m_score.notna()].copy()
risk["m_pct_in_sector"] = risk.groupby(["sector", "year"]).m_score.rank(pct=True)
risk["accruals_pct_in_sector"] = risk.groupby(["sector", "year"]).accruals.rank(pct=True)
print(f"Tỷ lệ công ty–năm có M-Score > −1,78: {(risk.m_score > -1.78).mean():.1%}")
cols = ["m_score", "m_pct_in_sector", "accruals", "accruals_pct_in_sector", "dsri", "gmi", "aqi", "depi", "sgai", "lvgi"]
display(risk[risk.ticker == FOCUS].set_index("year")[cols].T)

# %% [markdown]
# ## 8. Lưu artifact
#
# App sẽ đọc các file này thay vì train lại. Trên Kaggle chúng nằm ở `/kaggle/working/artifacts` (tab Output).

# %%
final_model.get_booster().save_model(str(OUT_DIR / "growth_model.json"))

pred_cols = ["ticker", "year", "sector", "exchange", "revenue", "pred_rel", "vol_group", "radius", "growth_lo", "growth_mid", "growth_hi", "revenue_next_mid"]
latest[[c for c in pred_cols if c in latest]].sort_values("pred_rel", ascending=False).to_csv(OUT_DIR / f"predictions_{latest_year + 1}.csv", index=False)

risk_cols = ["ticker", "year", "sector", "m_score", "m_pct_in_sector", "accruals", "accruals_pct_in_sector", "dso_chg", "recv_minus_rev_g"]
risk[risk_cols].to_csv(OUT_DIR / "risk_scores.csv", index=False)
imp_summary.sort_values("mô hình chính", ascending=False).to_csv(OUT_DIR / "shap_global.csv")
if local_summary is not None:
    local_summary.to_csv(OUT_DIR / f"shap_{FOCUS}_{latest_year + 1}.csv")

xgb_test = test_scores.loc["XGBoost"]
metrics = {
    "target": "log revenue growth t→t+1 minus the cross-sectional median of that year, clipped at ±%.1f" % CLIP,
    "features": FEATURES,
    "feature_labels": {f: FEATURE_LABELS.get(f, f) for f in FEATURES},
    "params": {**PARAMS, "rounds": ROUNDS, **{k: v for k, v in BASE.items() if k not in ("n_jobs", "random_state")}},
    "rows": {"dev": len(dev), "test": len(test), "final_fit": len(data), "predicted": len(latest)},
    "test": {"mae": float(xgb_test["MAE"]), "mae_market_baseline": float(test_scores.loc["Bằng mặt bằng chung", "MAE"]),
             "improvement": float(xgb_test["cải thiện so với mặt bằng chung"]), "ic": float(xgb_test["IC trung bình"])},
    "walk_forward": walk.reset_index().rename(columns={"năm t": "year", "số công ty": "n", "MAE mặt bằng chung": "mae_market", "MAE XGBoost": "mae_xgb", "cải thiện": "improvement"}).to_dict("records"),
    "interval": {"coverage_target": COVERAGE, "group_by": "rev_g_3y_std", "bins": VOL_BINS[1:-1], "labels": order,
                 "radius_log": {g: float(RADIUS[g]) for g in order if g in RADIUS}, "coverage_on_test": covered},
    "market_assumption_log": MARKET_ASSUMPTION,
    "shap_stability": stability,
    "base_value": base_f if len(row) else None,
}
(OUT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
print("Đã lưu:", sorted(p.name for p in OUT_DIR.iterdir()))

# %% [markdown]
# ## Đọc kết quả thế nào
#
# - **Mô hình xếp hạng được, nhưng không đoán được con số.** IC dương nghĩa là nó phân biệt được nhóm tăng nhanh với nhóm tăng chậm (xem bảng backtest và biểu đồ 5 nhóm). Nhưng MAE chỉ nhỉnh hơn baseline "ai cũng tăng bằng mặt bằng chung" vài phần trăm, và khoảng dự báo 80% rất rộng ngay cả với nhóm ổn định nhất. Tăng trưởng doanh thu một năm tới phần lớn không nằm trong BCTC năm nay.
# - **Baseline "giữ nguyên đà" thua xa.** Công ty vừa tăng mạnh thường tăng chậm lại, nên kéo dài xu hướng là cách dự báo tệ nhất ở đây.
# - **Con số tuyệt đối phụ thuộc vào giả định mặt bằng chung.** Mô hình chỉ nói công ty lệch bao nhiêu so với số đông.
# - **Giới hạn dữ liệu:** số liệu đến từ OCR; các dòng không khớp đẳng thức kế toán đã bị loại, nhưng lỗi ở dòng không có đẳng thức kiểm tra thì vẫn lọt. Bộ dữ liệu thiếu một số mã (ví dụ CMG) và không có ngân hàng, chứng khoán, bảo hiểm trong mô hình.
#
# Trong báo cáo của agent, mục Forecast nên ghi mức tin cậy **thấp** và luôn kèm khoảng dự báo.
