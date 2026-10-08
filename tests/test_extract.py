from pathlib import Path

import pytest

from finresearch.build import identity_errors
from finresearch.extract import extract_file, extract_report, parse_number

RAW = Path(__file__).resolve().parents[1] / "data/raw/ocr_annual_financials"


@pytest.mark.parametrize("cell, expected", [
    ("45.535.942.846.453", 45535942846453.0),
    ("53,263,331,043", 53263331043.0),
    ("(265.267.561.842)", -265267561842.0),
    ("-", 0.0),
    ("", None),
    ("V.1", None),
    ("12.34.567", None),
])
def test_parse_number(cell, expected):
    assert parse_number(cell) == expected


def _table(header, rows):
    tr = lambda cells: "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
    return "<table>" + tr(header) + "".join(tr(r) for r in rows) + "</table>"


BALANCE = [("100", "A. TÀI SẢN NGẮN HẠN", "600", "500"), ("200", "B. TÀI SẢN DÀI HẠN", "400", "300"),
           ("270", "TỔNG CỘNG TÀI SẢN", "1.000", "800"), ("300", "C. NỢ PHẢI TRẢ", "700", "500"),
           ("400", "D. VỐN CHỦ SỞ HỮU", "300", "300"), ("440", "TỔNG CỘNG NGUỒN VỐN", "1.000", "800")]


@pytest.mark.parametrize("layout", ["code_first", "label_first", "split_label"])
def test_code_column_is_found_in_any_position(layout):
    scale = lambda v: v + ".000.000"  # amounts must look like amounts, not note references
    if layout == "code_first":
        header = ["Mã số", "TÀI SẢN", "Thuyết minh", "2024 VND", "2023 VND"]
        rows = [[c, l, "", scale(a), scale(b)] for c, l, a, b in BALANCE]
    elif layout == "label_first":
        header = ["CHỈ TIÊU", "Mã số", "Thuyết minh", "Số cuối năm", "Số đầu năm"]
        rows = [[l, c, "5", scale(a), scale(b)] for c, l, a, b in BALANCE]
    else:
        header = ["", "CHỈ TIÊU", "Mã số", "Thuyết minh", "Số cuối năm", "Số đầu năm"]
        rows = [["I.", l, c, "V.1", scale(a), scale(b)] for c, l, a, b in BALANCE]
    report = extract_report("BẢNG CÂN ĐỐI KẾ TOÁN\nĐơn vị tính: VND\n" + _table(header, rows))
    bs = report.statements["BS"]
    assert bs.get("270", 0) == 1_000_000_000 and bs.get("270", 1) == 800_000_000
    assert bs.lines["400"][0].endswith("VỐN CHỦ SỞ HỮU")
    assert identity_errors(report)["err_balance"] == 0


def test_prior_year_on_the_left_is_swapped():
    header = ["CHỈ TIÊU", "Mã số", "Số đầu năm", "Số cuối năm"]
    rows = [[l, c, b + ".000.000", a + ".000.000"] for c, l, a, b in BALANCE]
    bs = extract_report(_table(header, rows)).statements["BS"]
    assert bs.get("270", 0) == 1_000_000_000


@pytest.mark.skipif(not (RAW / "FPT").exists(), reason="dataset not downloaded")
def test_fpt_2024_matches_audited_figures():
    path = next((RAW / "FPT/2024").glob("*Hopnhat*/*_extracted.txt"))
    report = extract_file(path)
    assert round(report.statements["IS"].get("10", 0) / 1e9) == 62849
    assert round(report.statements["IS"].get("10", 1) / 1e9) == 52618
    assert round(report.statements["IS"].get("60", 0) / 1e9) == 9427
    errs = identity_errors(report)
    assert errs["err_balance"] == 0 and errs["err_gross"] == 0 and errs["err_cash"] == 0
