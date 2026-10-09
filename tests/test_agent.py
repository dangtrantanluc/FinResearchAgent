import pytest

from finresearch.agent.render import render
from finresearch.agent.schema import Claim, Intent, Report, Section, Verdict, Verdicts
from finresearch.agent.validate import deterministic_problems, for_judge, validate
from finresearch.rag.search import Evidence
from finresearch.tools.metrics import Metric


def _metric(name, period, value, unit="pct"):
    return Metric(id=f"X.{name}.{period}", ticker="X", name=name, period=str(period), value=value, unit=unit, label=name, source="s")


@pytest.fixture
def state():
    metrics = [_metric("revenue", 2023, 52.6e12, "vnd"), _metric("revenue", 2024, 62.8e12, "vnd"),
               _metric("rev_g", 2022, 0.30), _metric("rev_g", 2023, -0.05), _metric("rev_g", 2024, 0.194)]
    evidence = Evidence(id="X-AR2024-p049-c1", ticker="X", year=2024, page=49, section="ĐÁNH GIÁ CHUNG", score=1.0,
                        text="Doanh thu dịch vụ CNTT nước ngoài tăng trưởng 27,4%, thị trường Nhật Bản tăng 32,2%.")
    return {"intent": Intent(tickers=["X"], start_year=2023, end_year=2024, language="vi"),
            "metrics": {m.id: m for m in metrics}, "evidence": {evidence.id: evidence}, "evidence_topics": {evidence.id: "revenue"}, "notes": []}


@pytest.mark.parametrize("text, evidence_ids, problem", [
    ("Doanh thu đạt {{X.revenue.2024}}.", [], None),
    ("Doanh thu đạt {{X.revenue.2025}}.", [], "unknown metric"),
    ("Doanh thu đạt 62.849 tỷ đồng.", [], "number 62.849"),                                        # typed, not referenced
    ("Năm 2024 doanh thu đạt {{X.revenue.2024}}.", [], None),                                       # a year is not a figure
    ("Thị trường Nhật Bản tăng 32,2%.", ["X-AR2024-p049-c1"], None),                               # printed in the cited passage
    ("Japan grew 32.2%.", ["X-AR2024-p049-c1"], None),                                              # same figure, English separators
    ("Thị trường Nhật Bản tăng 35,0%.", ["X-AR2024-p049-c1"], "number 35,0"),                      # not what the passage says
    ("Thị trường Nhật Bản tăng mạnh.", ["X-AR2024-p999-c1"], "unknown evidence"),
    ("Triển vọng rất tích cực.", [], "no metric and no evidence"),
    ("Doanh thu tăng {{X.rev_g.2024}}.", [], None),
    ("Doanh thu giảm {{X.rev_g.2024}}.", [], "contradicts"),                                        # the value is positive
    ("Revenue increased {{X.rev_g.2023}}.", [], "contradicts"),                                     # the value is negative
    ("Doanh thu thay đổi {{X.rev_g.2023}}.", [], None),
    ("Doanh thu tăng từ {{X.revenue.2023}} lên {{X.revenue.2024}}.", [], None),
    ("Doanh thu giảm từ {{X.revenue.2023}} xuống {{X.revenue.2024}}.", [], "contradicts their values"),
    ("Doanh thu đạt {{X.revenue.2023}}, sau đó giảm nhẹ xuống {{X.revenue.2024}}.", [], "contradicts their values"),
    ("Revenue went from {{X.revenue.2023}} to {{X.revenue.2024}}.", [], None),
    ("Doanh thu đạt {{X.revenue.2024}} năm 2024, tăng so với mức {{X.revenue.2023}} năm 2023.", [], None),    # later year first
    ("Doanh thu đạt {{X.revenue.2024}} năm 2024, giảm so với mức {{X.revenue.2023}} năm 2023.", [], "contradicts their values"),
    ("Revenue was {{X.revenue.2024}} in 2024, down from {{X.revenue.2023}}.", [], "contradicts their values"),
    ("Doanh thu {{X.revenue.2023}} năm 2023 và {{X.revenue.2024}} năm 2024.", [], None),             # no direction claimed
    ("Tăng trưởng doanh thu năm 2023 ở mức {{X.rev_g.2023}}.", [], None),                           # "tăng trưởng" is a noun
    ("Revenue growth was {{X.rev_g.2023}} in 2023.", [], None),                                     # so is "growth"
    ("Doanh thu tăng {{X.rev_g.2022}} năm 2022 và tiếp tục tăng {{X.rev_g.2024}} năm 2024.", [], None),  # slower growth is still growth
])
def test_deterministic_checks(state, text, evidence_ids, problem):
    problems = deterministic_problems(text, evidence_ids, state)
    assert (problems == []) if problem is None else any(problem in p for p in problems), problems


class FakeLLM:
    """Returns canned objects by schema; records what it was asked."""

    def __init__(self, **replies):
        self.replies, self.calls = replies, []

    def generate(self, schema, system, prompt, tier="fast"):
        self.calls.append((schema.__name__, tier, prompt))
        return self.replies[schema.__name__]


DRAFT = Report(sections=[
    Section(key="revenue", claims=[
        Claim(text="Doanh thu năm 2024 đạt {{X.revenue.2024}}, tăng {{X.rev_g.2024}}."),
        Claim(text="Động lực chính là thị trường Nhật Bản, tăng 32,2%.", evidence_ids=["X-AR2024-p049-c1"]),
        Claim(text="Công ty dẫn đầu thị trường Đông Nam Á.", evidence_ids=["X-AR2024-p049-c1"]),
        Claim(text="Doanh thu đạt 70.000 tỷ đồng."),
    ]),
])


JUDGED = Verdicts(items=[Verdict(index=0, verdict="supported", reason="figures only"),
                         Verdict(index=1, verdict="supported", reason="stated"),
                         Verdict(index=2, verdict="unsupported", reason="not in the passage")])


def test_the_judge_sees_each_figure_with_its_meaning(state):
    assert for_judge("Doanh thu đạt {{X.revenue.2024}}, tăng {{X.rev_g.2024}}.", state) == \
        "Doanh thu đạt ⟦62.800 tỷ đồng = revenue, 2024⟧, tăng ⟦19,4% = rev_g, 2024⟧."


def test_judge_verdicts_decide_which_claims_survive(state):
    llm = FakeLLM(Verdicts=JUDGED)
    out = validate({**state, "draft": DRAFT}, llm)
    kept = [c.kept for c in out["checked"]]
    assert kept == [True, True, False, False]
    assert out["stats"] == {"claims": 4, "kept": 2, "dropped_deterministic": 1, "dropped_unsupported": 1,
                            "qualitative": 2, "supported": 1, "partial": 0, "unchecked": 0}
    prompt = llm.calls[0][2]
    assert len(llm.calls) == 1 and "70.000" not in prompt               # a claim that already failed is not sent to the judge
    assert "CLAIM 0: Doanh thu năm 2024 đạt ⟦62.800 tỷ đồng = revenue, 2024⟧" in prompt and "PASSAGES:\n  none. This claim rests on its bracketed figures alone." in prompt


def test_without_a_model_cited_claims_are_kept_but_marked_unchecked(state):
    out = validate({**state, "draft": DRAFT}, None)
    assert [c.support for c in out["checked"]] == ["none_needed", "unchecked", "unchecked", "none_needed"]
    assert out["stats"]["kept"] == 3 and out["stats"]["unchecked"] == 2


def test_render_fills_numbers_and_numbers_the_citations(state):
    checked = validate({**state, "draft": DRAFT}, FakeLLM(Verdicts=JUDGED))
    md = render({**state, **checked})["markdown"]
    assert "Doanh thu năm 2024 đạt 62.800 tỷ đồng, tăng 19,4%." in md
    assert "tăng 32,2%. [1]" in md and "- [1] BCTN X 2024, tr. 49" in md
    assert "Đông Nam Á" not in md and "70.000" not in md               # dropped claims never reach the reader
    assert "{{" not in md
    assert "1/2 nhận định" in md and "Đã loại:** 2" in md


def _db_ready() -> bool:
    try:
        from finresearch.db import query
        return int(query("SELECT count(*) AS n FROM chunks WHERE ticker = 'FPT'").n.iloc[0]) > 0
    except Exception:
        return False


needs_db = pytest.mark.skipif(not _db_ready(), reason="Postgres not loaded or reports not ingested")


@needs_db
def test_offline_run_answers_the_example_question():
    from finresearch.agent.graph import run

    final = run("Phân tích kết quả kinh doanh của FPT giai đoạn 2022–2025 và ước tính doanh thu 2026", llm=None, search_mode="lexical")
    intent = final["intent"]
    assert (intent.tickers, intent.start_year, intent.end_year, intent.language) == (["FPT"], 2022, 2025, "vi")
    assert final["stats"]["claims"] == final["stats"]["kept"] > 15       # template sentences all pass their own validator
    md = final["markdown"]
    assert "70.113 tỷ đồng" in md and "16,8%" in md and "FPT Telecom" in md and "{{" not in md
    assert "Đoạn trích liên quan" in md and "BCTN FPT 2025, tr." in md


@needs_db
def test_comparison_puts_both_companies_in_one_table():
    from finresearch.agent.graph import run

    final = run("So sánh FPT và ELC giai đoạn 2023–2025", llm=None, search_mode="lexical")
    md = final["markdown"]
    table = md[md.index("## So sánh nhanh"): md.index("## Tóm tắt")]
    assert "| Chỉ tiêu | FPT | ELC |" in table
    assert "| Doanh thu thuần 2025 | 70.113 tỷ đồng | 1.518 tỷ đồng |" in table
    roe = next(l for l in table.splitlines() if l.startswith("| ROE 2025"))
    assert "28,3% (phân vị 100)" in roe and roe.count("phân vị") == 2      # each firm ranked within its own sector
    forecast = next(l for l in table.splitlines() if l.startswith("| Doanh thu dự báo 2026"))
    assert "55.549 tỷ đồng (44.123 tỷ đồng – 69.932 tỷ đồng)" in forecast


@needs_db
def test_single_company_report_has_no_comparison_table():
    from finresearch.agent.graph import run

    assert "So sánh nhanh" not in run("Phân tích FPT 2024-2025", llm=None, search_mode="lexical")["markdown"]


class DownLLM:
    """Every model out of quota."""

    calls: list = []

    def generate(self, schema, system, prompt, tier="fast"):
        raise RuntimeError("Gemini không trả lời được (model-a: 429)")


@needs_db
def test_a_model_outage_still_yields_a_report_and_says_so():
    from finresearch.agent.graph import run

    final = run("Phân tích FPT 2024-2025", llm=DownLLM(), search_mode="lexical")
    assert final["stats"]["kept"] == final["stats"]["claims"] > 10
    assert "70.113 tỷ đồng" in final["markdown"]
    assert "Mô hình ngôn ngữ không trả lời" in final["markdown"] and "model-a: 429" in final["markdown"]


@needs_db
def test_unknown_ticker_stops_before_any_work():
    from finresearch.agent.graph import run

    final = run("Phân tích ZZZ năm 2024", llm=None, search_mode="lexical")
    assert "error" in final and "markdown" not in final


@needs_db
def test_llm_draft_goes_through_the_validator():
    from finresearch.agent.graph import run

    draft = Report(sections=[Section(key="revenue", claims=[
        Claim(text="Doanh thu thuần năm 2025 đạt {{FPT.revenue.2025}}."),
        Claim(text="Doanh thu thuần năm 2025 đạt 70.113 tỷ đồng."),
        Claim(text="Doanh thu thuần năm 2025 đạt {{FPT.revenue.2031}}."),
    ])])
    llm = FakeLLM(Report=draft, Verdicts=Verdicts(items=[Verdict(index=0, verdict="supported", reason="figures only")]))
    final = run("Phân tích FPT 2024-2025", llm=llm, search_mode="lexical")
    assert [c.kept for c in final["checked"]] == [True, False, False]
    assert [name for name, _, _ in llm.calls] == ["Report", "Verdicts"]
    schema, tier, prompt = llm.calls[0]
    assert (schema, tier) == ("Report", "strong")
    assert "FPT.revenue.2025 | Doanh thu thuần | 70.113 tỷ đồng" in prompt and "[FPT-AR2025-p" in prompt
