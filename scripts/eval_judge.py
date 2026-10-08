"""Plant true and false claims and see which ones the judge model lets through.

    python scripts/eval_judge.py

One model call. The retrieval and numbers are real (FPT, 2022-2025); only the claims are made up.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finresearch.agent import nodes  # noqa: E402
from finresearch.agent.schema import Claim, Report, Section  # noqa: E402
from finresearch.agent.validate import validate  # noqa: E402
from finresearch.env import load_env  # noqa: E402
from finresearch.llm import default_llm  # noqa: E402

PLAN = "FPT-AR2025-p055-t1"  # the page with the company's 2026 targets
# (should the claim survive?, what it tests, text, cites the plan page?)
PLANTED = [
    (True, "kế hoạch đúng như in trong báo cáo", "Kế hoạch năm 2026 của FPT đặt mục tiêu doanh thu 58.580 tỷ, tăng 15,8% so với cùng kỳ.", True),
    (True, "chỉ nhắc lại hai số liệu", "Doanh thu thuần tăng từ {{FPT.revenue.2022}} năm 2022 lên {{FPT.revenue.2025}} năm 2025.", False),
    (True, "phép so sánh giữa hai số liệu", "Lợi nhuận sau thuế năm 2025 là {{FPT.net_income.2025}}, cao hơn mức {{FPT.net_income.2024}} của năm 2024.", False),
    (False, "nguyên nhân bịa", "Doanh thu năm 2025 tăng chủ yếu nhờ FPT mua lại một ngân hàng số tại Singapore.", True),
    (False, "ngược với kế hoạch", "Kế hoạch năm 2026 của FPT dự kiến doanh thu giảm so với năm 2025.", True),
    (False, "số liệu dùng sai nghĩa", "Khối Công nghệ đóng góp {{FPT.revenue_cagr.2022_2025}} tổng doanh thu của Tập đoàn.", True),
    (False, "khẳng định vượt ra ngoài số liệu, không có đoạn trích", "Biên lợi nhuận gộp {{FPT.gross_margin.2025}} là cao nhất trong số các công ty công nghệ Đông Nam Á.", False),
    (False, "rủi ro không có trong đoạn trích", "FPT đang bị cơ quan thuế điều tra về chuyển giá.", True),
]

if __name__ == "__main__":
    load_env(ROOT)
    llm = default_llm()
    if llm is None:
        raise SystemExit("Cần GEMINI_API_KEY trong .env")
    question = "Phân tích FPT giai đoạn 2022–2025 và ước tính doanh thu 2026"
    state = {"question": question, "intent": nodes.parse_intent(question)}
    for step in (nodes.gather, nodes.plan_queries, nodes.retrieve):
        state.update(step(state))
    claims = [Claim(text=text, evidence_ids=[PLAN] if cited else []) for _, _, text, cited in PLANTED]
    checked = validate({**state, "draft": Report(sections=[Section(key="revenue", claims=claims)])}, llm)["checked"]
    right = 0
    for (should_keep, what, _, _), c in zip(PLANTED, checked):
        ok = c.kept == should_keep
        right += ok
        print(f"{'ĐÚNG' if ok else 'SAI '} | {'giữ ' if c.kept else 'loại'} | {what} | {c.support}: {c.reason[:90]}")
    print(f"\n{right}/{len(PLANTED)} đúng | model chấm: {[c['model'] for c in llm.calls if 'error' not in c]}")
