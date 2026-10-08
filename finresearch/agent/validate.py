"""Check every claim before it reaches the reader.

Deterministic checks come first and cost nothing: a claim that cites a metric
or passage that does not exist, types a number that is in neither the metrics
nor its evidence, or says "rose" about a figure that fell is dropped. Claims that
survive go to the model in one call. It sees every figure with its meaning, so it
can tell a growth rate presented as a market share, and it judges whether the
cited passages actually say what the claim says.
"""
from __future__ import annotations

import re

from ..llm import LLM
from ..tools.metrics import format_metric
from .schema import PLACEHOLDER, CheckedClaim, Report, State, Verdicts

NUMBER = re.compile(r"\d[\d.,]*\d|\d")
# Verbs only. "tăng trưởng" and "growth" are nouns: "growth of -16%" is a correct sentence.
UP = r"tăng(?!\s+trưởng)|increas(?:ed|es|ing)|grew|grows|growing|rose|rises|rising|up"
DOWN = r"giảm|decreas(?:ed|es|ing)|declin(?:ed|es|ing)|fell|falls|falling|dropp?(?:ed|s|ing)|down"
DIRECTION = re.compile(rf"\b({UP}|{DOWN})\b[^.{{}}]{{0,60}}$", re.I)
GROWTH_NAMES = re.compile(r"(_g|_cagr|^forecast_growth_\w+)$")

BETWEEN = re.compile(rf"\b(?:({UP}|lên)|({DOWN}|xuống))\b", re.I)

# Said at the claim itself, not only in the system prompt: a small model otherwise answers
# "unsupported, no passages" to a sentence that merely restates two verified figures.
NO_PASSAGES = ("  none. This claim rests on its bracketed figures alone. Answer supported if it only states, compares or "
               "does arithmetic on them; answer unsupported if it asserts anything else.")

JUDGE_SYSTEM = """You check an analyst's claims before publication.
Figures in ⟦double brackets⟧ are written as ⟦value = what the value is⟧. They come from audited statements and are
correct; never look for them in the passages. Everything else in a claim must come from the passages listed under it.

For each numbered claim answer:
- supported: every bracketed figure is used as what it is, the wording agrees with the figures, and anything the claim
  says beyond the figures (a cause, a segment, a plan, a risk, another number) is stated in its passages. A claim that
  only restates bracketed figures, including whether they rose or fell, is supported even with no passages. So is
  arithmetic between bracketed figures: which is larger, a difference, a ratio, or one figure being part of another.
- partial: the main point holds but one detail is not in the passages. Also answer partial, and name both values, when
  a passage gives a different value for the same quantity as a bracketed figure (companies often compute a ratio
  differently); that difference alone never makes a claim unsupported.
- unsupported: the passages do not say it or say something different; or a bracketed figure is used as something else
  (for example a growth rate presented as a share of revenue); or the wording contradicts the figures.
Judge only from the brackets and the passages, not from what you know. Give a verdict for every claim."""


def _swap(token: str) -> str:
    return token.translate(str.maketrans(",.", ".,"))


def deterministic_problems(text: str, evidence_ids: list[str], state: State) -> list[str]:
    metrics, evidence = state["metrics"], state.get("evidence", {})
    problems = []
    for mid in PLACEHOLDER.findall(text):
        if mid not in metrics:
            problems.append(f"unknown metric {mid}")
    missing = [e for e in evidence_ids if e not in evidence]
    if missing:
        problems.append(f"unknown evidence {', '.join(missing)}")

    cited = " ".join(evidence[e].text for e in evidence_ids if e in evidence)
    for token in NUMBER.findall(PLACEHOLDER.sub(" ", text)):
        if re.fullmatch(r"(19|20)\d\d", token):
            continue
        if token not in cited and _swap(token) not in cited:
            problems.append(f"number {token} is neither a metric reference nor in the cited passages")

    for match in PLACEHOLDER.finditer(text):
        metric = metrics.get(match.group(1))
        if metric is None or not GROWTH_NAMES.search(metric.name):
            continue
        word = DIRECTION.search(text[: match.start()])
        if word:
            says_up = re.fullmatch(UP, word.group(1), re.I) is not None
            if (says_up and metric.value < 0) or (not says_up and metric.value > 0):
                problems.append(f"'{word.group(1)}' contradicts {metric.id} = {metric.value:+.3f}")
    # The same line item in two years with "rose" or "fell" between them: whichever is mentioned first
    # ("from A up to B", or "B, down from A"), the word describes the move from the earlier year to the later.
    refs = [(m, metrics.get(m.group(1))) for m in PLACEHOLDER.finditer(text)]
    for (m1, a), (m2, b) in zip(refs, refs[1:]):
        if a is None or b is None or (a.ticker, a.name) != (b.ticker, b.name):
            continue
        if not (a.period.isdigit() and b.period.isdigit()) or a.period == b.period:
            continue
        if GROWTH_NAMES.search(a.name):  # "rose 5%, then rose 3%" is about the level, not the rate; the sign check covers rates
            continue
        earlier, later = sorted((a, b), key=lambda m: int(m.period))
        words = BETWEEN.findall(text[m1.end(): m2.start()])
        ups, downs = sum(bool(u) for u, _ in words), sum(bool(d) for _, d in words)
        if (ups and not downs and later.value < earlier.value) or (downs and not ups and later.value > earlier.value):
            problems.append(f"wording between {a.id} and {b.id} contradicts their values")
    if not PLACEHOLDER.search(text) and not evidence_ids:
        problems.append("no metric and no evidence behind this claim")
    return problems


def for_judge(text: str, state: State) -> str:
    """The claim with each reference replaced by its value and meaning."""
    metrics, lang = state["metrics"], state["intent"].language

    def show(match):
        m = metrics[match.group(1)]
        return f"⟦{format_metric(m, lang)} = {m.label}, {m.period.replace('_', '–')}⟧"

    return PLACEHOLDER.sub(show, text)


def validate(state: State, llm: LLM | None = None) -> dict:
    report: Report = state["draft"]
    checked = [CheckedClaim(section=s.key, claim=c, problems=deterministic_problems(c.text, c.evidence_ids, state))
               for s in report.sections for c in s.claims]

    to_judge = [c for c in checked if not c.problems]
    if llm is None:
        for c in to_judge:  # figures still stand on the deterministic checks; cited claims are flagged as unjudged
            c.support = "unchecked" if c.claim.evidence_ids else "none_needed"
    elif to_judge:
        evidence = state["evidence"]
        blocks = []
        for n, c in enumerate(to_judge):
            passages = "\n".join(f"  [{e}] {evidence[e].text}" for e in c.claim.evidence_ids) or NO_PASSAGES
            blocks.append(f"CLAIM {n}: {for_judge(c.claim.text, state)}\nPASSAGES:\n{passages}")
        verdicts = {v.index: v for v in llm.generate(Verdicts, JUDGE_SYSTEM, "\n\n".join(blocks), tier="fast").items}
        for n, c in enumerate(to_judge):
            v = verdicts.get(n)
            c.support, c.reason = (v.verdict, v.reason) if v else ("unchecked", "the judge returned no verdict for this claim")

    cited = [c for c in checked if c.claim.evidence_ids and not c.problems]
    stats = {
        "claims": len(checked),
        "kept": sum(c.kept for c in checked),
        "dropped_deterministic": sum(bool(c.problems) for c in checked),
        "dropped_unsupported": sum(c.support == "unsupported" for c in checked),
        "qualitative": len(cited),
        "supported": sum(c.support == "supported" for c in cited),
        "partial": sum(c.support == "partial" for c in cited),
        "unchecked": sum(c.support == "unchecked" for c in cited),
    }
    return {"checked": checked, "stats": stats}
