"""The workflow as a LangGraph graph.

    python -m finresearch.agent.graph "Analyze FPT's financial performance from 2022-2025 and estimate 2026 revenue"

intent → gather → plan_queries → retrieve → draft → validate → render. The path is
fixed on purpose: every question in this domain needs the same steps, so a planner
would add variance without adding ability. A model is used in two places, to write
the draft and to judge its qualitative claims; without one the graph still runs,
writing from templates and leaving cited claims marked as unchecked.
"""
from __future__ import annotations

import sys

from langgraph.graph import END, START, StateGraph

from ..llm import LLM, default_llm
from ..rag.search import Mode
from . import nodes
from .draft import llm_draft, template_draft
from .render import render
from .schema import Intent, State
from .validate import validate


def build_graph(llm: LLM | None = None, search_mode: Mode = "hybrid"):
    def intent(state: State) -> dict:
        if hasattr(llm, "calls"):
            llm.calls.clear()
        result = nodes.parse_intent(state["question"], llm)
        return {"intent": result} if isinstance(result, Intent) else {"error": result}

    def draft(state: State) -> dict:
        if llm is None:
            return {"draft": template_draft(state)}
        try:
            return {"draft": llm_draft(state, llm)}
        except (RuntimeError, ValueError) as exc:  # every model out of quota or overloaded, or no valid JSON
            english = state["intent"].language == "en"
            note = (f"The language model did not answer, so this note is written from fixed templates and has no qualitative statements. ({exc})"
                    if english else
                    f"Mô hình ngôn ngữ không trả lời, nên báo cáo này viết theo mẫu câu cố định và không có nhận định định tính. ({exc})")
            return {"draft": template_draft(state), "notes": state.get("notes", []) + [note]}

    g = StateGraph(State)
    g.add_node("intent", intent)
    g.add_node("gather", nodes.gather)
    g.add_node("plan_queries", nodes.plan_queries)
    g.add_node("retrieve", lambda s: nodes.retrieve(s, mode=search_mode))
    g.add_node("draft", draft)
    g.add_node("validate", lambda s: validate(s, llm))
    g.add_node("render", lambda s: {**render(s), "llm_calls": list(getattr(llm, "calls", []))})
    g.add_edge(START, "intent")
    g.add_conditional_edges("intent", lambda s: END if s.get("error") else "gather")
    for a, b in [("gather", "plan_queries"), ("plan_queries", "retrieve"), ("retrieve", "draft"),
                 ("draft", "validate"), ("validate", "render"), ("render", END)]:
        g.add_edge(a, b)
    return g.compile()


def run(question: str, llm: LLM | None = None, search_mode: Mode = "hybrid") -> State:
    return build_graph(llm, search_mode).invoke({"question": question})


if __name__ == "__main__":
    from ..env import load_env

    load_env()
    final = run(" ".join(sys.argv[1:]), default_llm())
    print(final.get("error") or final["markdown"])
    if final.get("stats"):
        print("\n---", final["stats"], file=sys.stderr)
