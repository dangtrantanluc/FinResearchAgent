"""Streamlit front end.

    streamlit run app.py
"""
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from finresearch.agent.graph import build_graph
from finresearch.env import load_env, silence_source_watcher
from finresearch.llm import default_llm
from finresearch.tools.metrics import format_metric

ROOT = Path(__file__).resolve().parent
EXAMPLE = "Phân tích kết quả kinh doanh của FPT giai đoạn 2022–2025, chỉ ra động lực tăng trưởng, rủi ro tài chính và ước tính doanh thu 2026"
STEPS = {
    "intent": "Đọc câu hỏi",
    "gather": "Lấy số liệu, chỉ số, tín hiệu rủi ro và dự báo",
    "plan_queries": "Lập danh sách câu hỏi cho báo cáo thường niên",
    "retrieve": "Tìm đoạn trích",
    "draft": "Viết nháp",
    "validate": "Kiểm tra từng nhận định",
    "render": "Dựng báo cáo",
}

st.set_page_config(page_title="FinResearch Agent", layout="wide")
load_env(ROOT)
silence_source_watcher()  # .streamlit/config.toml turns the watcher off, but a reconnecting browser tab restarts it


@st.cache_resource
def workflow():
    llm = default_llm()
    return build_graph(llm), llm


graph, llm = workflow()

st.title("FinResearch Agent")
st.caption(
    f"Mô hình ngôn ngữ: Gemini. Viết nháp: {' → '.join(llm.models['strong'])}. Kiểm tra: {' → '.join(llm.models['fast'])}." if llm else
    "Chưa có GEMINI_API_KEY trong .env: báo cáo chỉ gồm số liệu viết theo mẫu, kèm đoạn trích nguyên văn."
)
question = st.text_area("Câu hỏi (viết mã chứng khoán bằng chữ in hoa)", EXAMPLE, height=80)

if st.button("Phân tích", type="primary") and question.strip():
    state = {"question": question.strip()}
    started = time.time()
    with st.status("Đang chạy…", expanded=True) as status:
        for update in graph.stream(state, stream_mode="updates"):
            for node, changes in update.items():
                state.update(changes or {})
                st.write(f"✓ {STEPS.get(node, node)}")
        status.update(label=f"Xong trong {time.time() - started:.0f} giây", state="complete", expanded=False)
    st.session_state["result"] = state

    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    record = {k: (v.model_dump() if hasattr(v, "model_dump") else v) for k, v in state.items() if k in ("question", "intent", "error", "notes", "flags", "queries", "markdown", "stats")}
    record["checked"] = [c.model_dump() for c in state.get("checked", [])]
    record["evidence"] = {eid: state.get("evidence_topics", {}).get(eid) for eid in state.get("evidence", {})}
    (runs / f"{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

result = st.session_state.get("result")
if result and result.get("error"):
    st.error(result["error"])
elif result:
    lang = result["intent"].language
    st.markdown(result["markdown"])

    s = result["stats"]
    cols = st.columns(4)
    cols[0].metric("Nhận định", s["claims"])
    cols[1].metric("Giữ lại", s["kept"])
    cols[2].metric("Loại vì sai số liệu hoặc thiếu nguồn", s["dropped_deterministic"])
    cols[3].metric("Loại vì đoạn trích không hỗ trợ", s["dropped_unsupported"])

    answered = [c for c in result.get("llm_calls", []) if "error" not in c]
    failed = [c for c in result.get("llm_calls", []) if "error" in c]
    if answered:
        st.caption("Model đã trả lời: " + "; ".join(f"{c['schema']} — {c['model']} ({c['seconds']} giây)" for c in answered)
                   + (f". Bỏ qua {len(failed)} lần gọi lỗi: " + "; ".join(f"{c['model']} ({c['error'][:3]})" for c in failed) if failed else ""))

    dropped = [c for c in result["checked"] if not c.kept]
    with st.expander(f"Nhận định bị loại ({len(dropped)})"):
        for c in dropped:
            st.markdown(f"- {c.claim.text}  \n  *{'; '.join(c.problems) or c.reason or c.support}*")
    partial = [c for c in result["checked"] if c.kept and c.support == "partial"]
    if partial:
        with st.expander(f"Nhận định chỉ được hỗ trợ một phần ({len(partial)})"):
            for c in partial:
                st.markdown(f"- {c.claim.text}  \n  *{c.reason}*")

    with st.expander(f"Số liệu đã dùng ({len(result['metrics'])})"):
        st.dataframe(pd.DataFrame([{"ID": m.id, "Chỉ tiêu": m.label, "Kỳ": m.period, "Giá trị": format_metric(m, lang), "Nguồn": m.source}
                                   for m in result["metrics"].values()]), hide_index=True, width="stretch")

    drivers = [m for m in result["metrics"].values() if m.name.startswith("driver_")]
    if drivers:
        with st.expander("Vì sao mô hình dự báo như vậy (SHAP)"):
            chart = pd.DataFrame({"Đóng góp (điểm %)": [m.value * 100 for m in drivers]},
                                 index=[f"{m.ticker}: {m.label.split(': ', 1)[-1]}" for m in drivers])
            st.bar_chart(chart, horizontal=True)

    with st.expander(f"Đoạn trích từ báo cáo thường niên ({len(result.get('evidence', {}))})"):
        for eid, e in result.get("evidence", {}).items():
            st.markdown(f"**{e.citation(lang)}** · {e.section} · `{eid}`")
            st.text(e.text)
