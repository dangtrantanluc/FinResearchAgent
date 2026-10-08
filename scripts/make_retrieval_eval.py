"""Rebuild eval/retrieval_fpt.jsonl: questions about FPT's annual reports with the pages that answer them.

Gold pages are every page of that year's PDF whose text matches `pattern`, a
regex for the fact that answers the question. Questions are paraphrased, and
half are in English, so a hit cannot come from copying the pattern.

    python scripts/make_retrieval_eval.py
"""
import json
import re
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]

QUESTIONS = [
    (2024, "vi", "Giá trị hợp đồng ký mới của mảng dịch vụ CNTT ở nước ngoài năm 2024 là bao nhiêu?", r"ký mới.{0,90}1,3 tỷ USD"),
    (2024, "en", "What revenue and profit targets did FPT set for 2025, and where does it plan to invest?", r"KẾ HOẠCH NĂM 2025"),
    (2024, "vi", "FPT đối phó với biến động tỷ giá hối đoái bằng những biện pháp nào?", r"rủi ro tỷ giá"),
    (2024, "en", "What does FPT's partnership with NVIDIA involve?", r"NVIDIA.{0,80}[Nn]hà máy AI|hợp tác.{0,40}NVIDIA"),
    (2024, "vi", "Mảng dịch vụ CNTT tại Nhật tăng trưởng ra sao trong năm 2024?", r"Nhật Bản.{0,140}(36,3%|32,2%)"),
    (2024, "en", "How many learners were enrolled across FPT's education system in 2024?", r"152\.000"),
    (2024, "vi", "FPT dành bao nhiêu phần lợi nhuận để chia cổ tức tiền mặt trong năm 2024?", r"Tỷ lệ lợi nhuận trả cổ tức|Cổ tức tiền mặt trả trong năm"),
    (2023, "en", "Which companies did FPT acquire or invest in during 2023?", r"Cardinal Peak"),
    (2023, "vi", "Năm 2023 doanh thu từ khách hàng nước ngoài của mảng dịch vụ CNTT chạm cột mốc nào?", r"(0?1|[Mm]ột) tỷ USD.{0,90}nước ngoài|nước ngoài.{0,90}(0?1|[Mm]ột) tỷ USD"),
    (2023, "en", "What were FPT's business targets for 2024?", r"KẾ HOẠCH NĂM 2024"),
    (2022, "vi", "Mảng chuyển đổi số mang về bao nhiêu doanh thu năm 2022?", r"7\.349"),
    (2022, "en", "How did the pay-TV business perform in 2022?", r"truyền hình trả tiền.{0,140}40%"),
    (2022, "vi", "FPT có những hành động gì để cắt giảm khí nhà kính?", r"phát thải khí nhà kính"),
    (2021, "en", "What strategic investment did FPT make in the SME software platform Base.vn?", r"đầu tư chiến lược vào.{0,80}Base\.vn"),
    (2021, "vi", "FPT đã chi bao nhiêu để hỗ trợ chống dịch trong năm 2021?", r"69,5 tỷ"),
    (2025, "en", "Why will FPT Telecom be accounted for under the equity method from 2026?", r"FPT Telecom.{0,140}(phương pháp vốn chủ sở hữu|Bộ Công an)"),
    (2025, "vi", "FPT đặt mục tiêu xuất khẩu bao nhiêu chip bán dẫn?", r"10 triệu chip"),
    (2025, "en", "What data center capacity did FPT bring into operation in 2025?", r"3\.600 tủ rack|Trung tâm dữ liệu quy mô 10\.000"),
    (2025, "vi", "Kế hoạch kinh doanh năm 2026 của FPT gồm những mục tiêu nào?", r"KẾ HOẠCH NĂM 2026"),
    (2023, "en", "What long-term revenue ambition does FPT have for its automotive software business?", r"01 tỷ USD.{0,30}2030|doanh thu 01 tỷ USD vào năm 2030"),
]


def gold_pages(year: int, pattern: str) -> list[int]:
    doc = pymupdf.open(ROOT / f"data/reports/FPT/FPT_{year}.pdf")
    return [i for i, p in enumerate(doc, 1) if re.search(pattern, re.sub(r"\s+", " ", p.get_text("text")))]


if __name__ == "__main__":
    rows = []
    for n, (year, lang, question, pattern) in enumerate(QUESTIONS, 1):
        pages = gold_pages(year, pattern)
        rows.append({"id": f"fpt-{n:02d}", "ticker": "FPT", "year": year, "lang": lang, "question": question,
                     "gold_pages": pages, "pattern": pattern})
        print(f"{n:2d} {year} {lang} gold={pages}  {question}")
    empty = [r["id"] for r in rows if not r["gold_pages"]]
    if empty:
        sys.exit(f"no gold page for: {empty}")
    out = ROOT / "eval/retrieval_fpt.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")
