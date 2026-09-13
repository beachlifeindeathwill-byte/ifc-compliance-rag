from __future__ import annotations

import json
import hashlib
import os
import re
import sys
from collections import Counter
from html import escape
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_faiss_vectorstore import load_env_file
from conversation_context import (
    ConversationState,
    has_followup_signal,
    is_meta_reference_question,
    merge_evidence_results,
    normalize_reference_context,
    prefers_context_evidence,
    rank_context_evidence,
    recent_evidence,
    remember_evidence,
    remember_question,
    rewrite_followup,
    update_state_from_results,
)
from context_router import ContextDecision, route_context_with_llm
from field_extraction import field_recall_status as evidence_field_recall_status
from hybrid_retrieval import hybrid_search, load_vectorstore
from ifc_fire_parser import build_ifc_review_questions, ifc_processing_chain, parse_ifc_file
from llm_answer import LLMAnswerError, generate_answer
from query_intent import detect_query_intent
from retrieval_policy import infer_route, is_abolished, load_policy


DEFAULT_ENV_FILE = ROOT / ".env"
RAW_PDF_DIR = ROOT / "data" / "raw_pdfs"
IFC_UPLOAD_DIR = ROOT / "data" / "ifc_uploads"

STANDARD_NAMES = {
    "GB 55037-2022": "建筑防火通用规范",
    "GB 55036-2022": "消防设施通用规范",
    "GB 50016-2014": "建筑设计防火规范",
    "GB 50067-2014": "汽车库、修车库、停车场设计防火规范",
}

QUESTION_TEMPLATES = [
    {
        "title": "数值要求",
        "hint": "适合净宽、面积、距离、耐火极限等问题",
        "query": "某类建筑中，某个部位的最小/最大数值要求是多少？",
    },
    {
        "title": "设置条件",
        "hint": "适合中庭、防火卷帘、消防设施、疏散设施",
        "query": "在某类建筑的某个场景下，某项防火设施应满足哪些设置要求？",
    },
    {
        "title": "合规判断",
        "hint": "适合给出一个设计值，让系统引用条文判断",
        "query": "某建筑场景下，某设计值为 X，是否满足规范要求？",
    },
    {
        "title": "跨条文复核",
        "hint": "适合旧规范与通用规范、专项规范之间的校核",
        "query": "某场景涉及旧版设计规范和现行通用规范时，应如何复核依据？",
    },
]

UI_BLOCKS = {
    "top_bar": {"label": "顶部模块栏", "order": 10, "title_size": 18, "body_size": 12, "width": "wide"},
    "qa_scope": {"label": "规范问答-知识库范围", "order": 20, "title_size": 14, "body_size": 13, "width": "wide"},
    "qa_intro": {"label": "规范问答-说明卡片", "order": 30, "title_size": 14, "body_size": 13, "width": "wide"},
    "qa_templates": {"label": "规范问答-问题模板", "order": 40, "title_size": 14, "body_size": 13, "width": "wide"},
    "qa_answers": {"label": "规范问答-对话回答", "order": 50, "title_size": 15, "body_size": 13, "width": "wide"},
    "qa_input": {"label": "规范问答-底部输入", "order": 60, "title_size": 14, "body_size": 13, "width": "wide"},
    "ifc_scope": {"label": "BIM审查-知识库范围", "order": 20, "title_size": 14, "body_size": 13, "width": "wide"},
    "ifc_panel": {"label": "BIM审查-模型工作台", "order": 30, "title_size": 15, "body_size": 13, "width": "wide"},
    "sidebar_library": {"label": "左侧-规范库状态", "order": 70, "title_size": 12, "body_size": 12, "width": "wide"},
    "sidebar_settings": {"label": "左侧-问答设置", "order": 80, "title_size": 12, "body_size": 12, "width": "wide"},
}


def ensure_env() -> None:
    if os.getenv("SILICONFLOW_API_KEY"):
        return
    if DEFAULT_ENV_FILE.exists():
        load_env_file(DEFAULT_ENV_FILE)


@st.cache_resource(show_spinner=False)
def get_policy():
    return load_policy()


def clean_text(text: str, limit: int | None = None) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit].strip() if limit else text


def load_eval_summary(filename: str) -> dict:
    path = ROOT / "data" / "retrieval_runs" / filename
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("summary", {})
    except (OSError, json.JSONDecodeError):
        return {}


def document_options(registry: dict, metas: list[dict] | None = None) -> list[dict]:
    standards = registry.get("standards", {})
    source_by_standard: dict[str, str] = {}
    for meta in metas or []:
        standard_id = meta.get("standard_id")
        source_file = meta.get("source_file")
        if standard_id and source_file and standard_id not in source_by_standard:
            source_by_standard[standard_id] = source_file

    options: list[dict] = [
        {
            "kind": "auto",
            "label": "自动识别（推荐）",
            "standard_id": "",
            "caption": "系统根据问题里的标准号、建筑场景和关键词自动判断检索范围。",
        },
        {
            "kind": "all",
            "label": "全库检索",
            "standard_id": "",
            "caption": "不指定单一文件，适合跨规范、跨条文或不确定适用范围的问题。",
        },
    ]

    for standard_id, item in sorted(
        standards.items(),
        key=lambda pair: (-int(pair[1].get("priority", 0)), pair[0]),
    ):
        source_file = source_by_standard.get(standard_id) or item.get("file") or f"{standard_id}.pdf"
        title = item.get("title") or STANDARD_NAMES.get(standard_id, standard_id)
        options.append(
            {
                "kind": "standard",
                "label": source_file,
                "standard_id": standard_id,
                "caption": f"{title} | {standard_id}",
            }
        )

    known_files = {item.get("file") for item in standards.values()}
    known_files.update(source_by_standard.values())
    for pdf_path in sorted(RAW_PDF_DIR.glob("*.pdf")):
        if pdf_path.name in known_files:
            continue
        options.append(
            {
                "kind": "file",
                "label": pdf_path.name,
                "standard_id": "",
                "caption": "已放入原始文件夹，但尚未在规范注册表中配置标准号和适用范围。",
            }
        )
    return options


def standard_label(standard_id: str) -> str:
    return f"{STANDARD_NAMES.get(standard_id, standard_id)}（{standard_id}）"


def exclusion_conflict_reason(question: str, text: str) -> str | None:
    compact_question = re.sub(r"\s+", "", question or "")
    compact_text = re.sub(r"\s+", "", text or "")
    if not compact_question or not compact_text:
        return None
    preferred_terms = [
        "中庭",
        "汽车库",
        "修车库",
        "地下",
        "半地下",
        "住宅建筑",
        "公共建筑",
        "医疗建筑",
        "厂房",
        "仓库",
        "防火卷帘",
        "防火墙",
    ]
    question_terms = [term for term in preferred_terms if term in compact_question]
    question_terms.extend(re.findall(r"[\u4e00-\u9fff]{2,8}", compact_question))

    for match in re.finditer(r"除([^。；;，,]{1,24})外", compact_text):
        excluded_scope = match.group(1)
        for term in question_terms:
            if excluded_scope in compact_question or excluded_scope in term:
                return f"首条证据包含排除适用范围“除{excluded_scope}外”，与问题中的“{term}”存在冲突。"

    for match in re.finditer(r"(?:不适用|不适用于|不应采用)([\u4e00-\u9fff]{1,16})", compact_text):
        excluded_scope = match.group(1)
        for term in question_terms:
            if excluded_scope in compact_question or excluded_scope in term:
                return f"首条证据包含“不适用/不应采用”类限制，与问题中的“{term}”存在冲突。"
    return None


def inject_css() -> None:
    st.markdown(
        """
<style>
div[data-testid="stAppViewContainer"] {
    background: #fbf9fa;
}
.stApp, body, button, input, textarea, select {
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Segoe UI", Arial, sans-serif !important;
}
header[data-testid="stHeader"],
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
#MainMenu {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}
section[data-testid="stSidebar"] {
    background: #22272a;
    border-right: 1px solid #3d4143;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 14px !important;
}
div[data-testid="stSidebarUserContent"] {
    padding: 14px 22px 22px 22px !important;
}
section[data-testid="stSidebar"] * {
    color: #e5edf7;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button {
    background: transparent;
    border: 1px solid transparent;
    color: #f8fafc;
    border-radius: 4px;
    justify-content: flex-start;
    padding-left: 12px;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
    background: #316bf3;
    border: 1px solid #316bf3;
    color: #ffffff;
}
.block-container {
    padding-top: 0;
    padding-bottom: 1.5rem;
    max-width: 100%;
}
.workbench-title {
    font-size: 17px;
    font-weight: 750;
    color: #172033;
    margin: 0 0 10px 0;
}
.soft-panel {
    background: rgba(255,255,255,0.96);
    border: 1px solid #d9e0ea;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
}
.section-card {
    background: #ffffff;
    border: 1px solid #d7dee9;
    border-radius: 8px;
    padding: 18px 20px;
    margin: 12px 0;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
}
.starter-card {
    border: 1px solid #d7dee9;
    border-radius: 6px;
    padding: 12px 14px;
    background: #ffffff;
    margin: 8px 0;
}
.starter-title {
    color: #172033;
    font-size: 14px;
    font-weight: 750;
    margin-bottom: 5px;
}
.starter-text {
    color: #64748b;
    font-size: 13px;
    line-height: 1.55;
}
.scope-strip {
    width: calc(100% - 96px);
    max-width: 1160px;
    margin: 2px 48px 16px 48px;
}
.mode-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 56px;
    padding: 0 28px;
    margin: 0 -1rem 18px -1rem;
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
}
.mode-nav-title {
    color: #1d2b3e;
    font-size: 18px;
    font-weight: 750;
    margin: 0;
}
.mode-nav-caption {
    color: #64748b;
    font-size: 12.5px;
    line-height: 1.4;
    margin-bottom: 0;
}
.top-actions {
    display: flex;
    align-items: center;
    gap: 10px;
}
.top-search {
    width: 260px;
    height: 34px;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    color: #64748b;
    display: flex;
    align-items: center;
    padding: 0 12px;
    font-size: 12px;
    background: #fbf9fa;
}
.top-chip {
    border: 1px solid #0051d5;
    color: #0051d5;
    background: #ffffff;
    border-radius: 4px;
    padding: 7px 12px;
    font-size: 12px;
    font-weight: 800;
}
.module-subtitle {
    color: #64748b;
    font-size: 13px;
    line-height: 1.6;
    width: calc(100% - 96px);
    max-width: 1160px;
    margin: -4px 48px 16px 48px;
}
.module-page {
    width: calc(100% - 96px);
    max-width: 1160px;
    margin: 0 48px;
}
.module-card {
    background: #ffffff;
    border: 1px solid #d7dee9;
    border-radius: 8px;
    padding: 16px 18px;
    margin: 12px 0;
}
div[data-testid="stSegmentedControl"] {
    max-width: 1180px;
    margin: 0 auto 24px auto;
}
div[data-testid="stSegmentedControl"] label {
    width: 100%;
}
div[data-testid="stSegmentedControl"] button {
    min-height: 52px;
    font-size: 15px;
    font-weight: 800;
}
.scope-card-title {
    color: #172033;
    font-size: 15px;
    font-weight: 800;
    margin-bottom: 4px;
}
.scope-card-caption {
    color: #64748b;
    font-size: 13px;
    line-height: 1.55;
    margin: 2px 0 8px 0;
}
.summary-card {
    border-left: 4px solid #1d5bd7;
}
.warning-card {
    background: #fff7ed;
    border: 1px solid #ff8f45;
    border-left: 4px solid #f97316;
}
.evidence-card {
    border: 1px solid #d7dee9;
    border-radius: 7px;
    padding: 14px;
    min-height: 118px;
    background: #ffffff;
}
.trace-card {
    background: #2b3d55;
    color: #e5edf7;
    border-radius: 7px;
    padding: 14px 16px;
    margin-top: 16px;
}
.trace-card code, .trace-card pre {
    color: #bff7d0;
    background: rgba(15, 23, 42, 0.22);
}
.chat-shell {
    width: calc(100% - 96px);
    max-width: 1160px;
    margin: 0 48px;
}
.chat-canvas {
    max-width: 900px;
    min-height: 540px;
    margin: 0 auto;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 20px 20px 14px 20px;
    box-shadow: 0 14px 34px rgba(15, 23, 42, 0.05);
}
.page-shell {
    max-width: 1180px;
    margin: 0 auto;
}
.top-tabs {
    max-width: 1180px;
    margin: 0 auto;
}
div[data-testid="stTabs"] {
    max-width: 1180px;
    margin: 0 auto;
}
div[data-testid="stTabs"] [role="tablist"] {
    gap: 0;
    border-bottom: 1px solid #d7dee9;
}
div[data-testid="stTabs"] button[role="tab"] {
    flex: 1 1 0;
    justify-content: center;
    min-height: 46px;
    border-radius: 0;
    border: 0;
    border-bottom: 2px solid transparent;
    background: #ffffff;
    font-size: 15px;
    font-weight: 750;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #0051d5;
    background: #ffffff;
    border-bottom-color: #0051d5;
}
div[data-testid="stTabs"] div[role="tabpanel"] {
    padding-top: 22px;
}
.query-input-panel {
    width: calc(100% - 96px);
    max-width: 1160px;
    margin: 18px 48px 0 48px;
    background: #ffffff;
    border: 1px solid #d7dee9;
    border-radius: 6px;
    padding: 10px 12px;
    box-shadow: none;
}
.user-turn-label {
    color: #64748b;
    font-size: 13px;
    margin: 20px 0 6px 0;
}
.user-bubble {
    margin-left: min(180px, 22%);
    background: #316bf3;
    color: #ffffff;
    border-radius: 6px;
    padding: 12px 16px;
    font-weight: 750;
    box-shadow: 0 8px 18px rgba(49, 107, 243, 0.14);
}
.assistant-card {
    border-left: 0;
    background: #ffffff;
    border: 1px solid #d7dee9;
    border-radius: 6px;
    padding: 18px 20px;
    margin: 10px 0 26px 0;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
}
.answer-main {
    border-left: 3px solid #22c55e;
    padding-left: 14px;
}
.answer-title {
    color: #0f172a;
    font-size: 15.5px;
    font-weight: 560;
    line-height: 1.78;
}
.answer-basis {
    color: #64748b;
    font-size: 13px;
    line-height: 1.75;
    margin-top: 8px;
}
.hl-scenario {
    color: #075985;
    font-weight: 650;
}
.hl-number {
    color: #a16207;
    background: #fffbeb;
    border-bottom: 1px solid #facc15;
    border-radius: 3px;
    padding: 0 2px;
    font-weight: 650;
}
.hl-source {
    color: #166534;
    background: #f0fdf4;
    border-bottom: 1px solid #86efac;
    border-radius: 3px;
    padding: 0 2px;
    font-weight: 650;
}
.hl-page {
    color: #7c3aed;
    background: #faf5ff;
    border-bottom: 1px solid #c4b5fd;
    border-radius: 3px;
    padding: 0 2px;
    font-weight: 650;
}
.answer-badge {
    display: inline-block;
    padding: 4px 9px;
    border-radius: 999px;
    background: #dcfce7;
    color: #166534;
    font-size: 12px;
    font-weight: 650;
}
.confidence-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    margin: 12px 0 14px 0;
}
.answer-badge.warn {
    background: #fee2e2;
    color: #991b1b;
}
.answer-badge.mid {
    background: #fef3c7;
    color: #92400e;
}
.answer-badge.model {
    background: #e0f2fe;
    color: #075985;
}
.answer-badge.model.warn {
    background: #f1f5f9;
    color: #475569;
}
div[data-testid="stExpander"] {
    margin-top: 14px;
}
div[data-testid="stExpander"] details summary p {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #334155 !important;
}
div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] p,
div[data-testid="stExpander"] div[data-testid="stCaptionContainer"] p,
div[data-testid="stExpander"] li {
    font-size: 13px !important;
    line-height: 1.65 !important;
}
.source-summary {
    margin-top: 16px;
    border: 1px solid #e2e8f0;
    border-radius: 7px;
    background: #fbfaf8;
    padding: 10px 14px;
    color: #b91c1c;
    font-weight: 750;
}
.source-mini {
    border: 1px solid #e2e8f0;
    border-radius: 7px;
    padding: 12px 14px;
    margin: 8px 0;
    background: #ffffff;
}
.source-meta {
    color: #64748b;
    font-size: 13px;
    margin: 6px 0;
}
.source-text {
    color: #1e293b;
    font-size: 13.5px;
    line-height: 1.75;
}
.understood {
    color: #64748b;
    font-size: 13px;
    margin: 8px 0 0 80px;
}
.mini-label {
    color: #64748b;
    font-size: 12px;
    font-weight: 700;
}
.pill {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 999px;
    background: #eef4ff;
    color: #1d4ed8;
    font-size: 12px;
    font-weight: 700;
}
.score-high { color: #059669; font-weight: 800; }
.score-mid { color: #d97706; font-weight: 800; }
.score-low { color: #dc2626; font-weight: 800; }
.sidebar-title {
    font-size: 19px;
    line-height: 1.25;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 4px 0;
}
.sidebar-version {
    color: #aaadaf;
    font-size: 11px;
    font-weight: 700;
    margin-bottom: 18px;
}
.sidebar-nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 11px;
    border-radius: 5px;
    color: #c4c7c9;
    font-size: 13px;
    font-weight: 800;
    margin-bottom: 6px;
}
.sidebar-nav-item.active {
    background: #316bf3;
    color: #ffffff;
}
.sidebar-section-label {
    margin-top: 16px;
    margin-bottom: 8px;
    color: #f8fafc;
    font-size: 12px;
    font-weight: 850;
}
.sidebar-doc {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 5px;
    padding: 9px 10px;
    margin: 7px 0;
    font-size: 12px;
}
.sidebar-kv {
    background: rgba(255,255,255,0.06);
    border-radius: 6px;
    padding: 10px 12px;
    margin-top: 14px;
    font-size: 13px;
}
.sidebar-alert {
    border-left: 3px solid #f59e0b;
    background: rgba(245,158,11,0.14);
    border-radius: 4px;
    padding: 10px 12px;
    margin-top: 18px;
    font-size: 13px;
}
div[data-testid="stTextInput"] input {
    border-radius: 6px;
    min-height: 42px;
}
div[data-testid="stButton"] button[kind="primary"] {
    min-height: 42px;
    border-radius: 6px;
    background: #075bd8;
}
div[data-testid="stButton"] button {
    font-size: 14px;
    font-weight: 650;
}
div[data-testid="stChatInput"] {
    max-width: 980px;
    margin: 0 auto;
}
div[data-testid="stBottomBlockContainer"],
div[data-testid="stChatFloatingInputContainer"] {
    display: none !important;
}
div[data-testid="stBottomBlockContainer"] > div {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


def confidence_class(level: str) -> str:
    return {"high": "score-high", "medium": "score-mid", "low": "score-low"}.get(level, "score-low")


def ui_key(block_id: str, field: str) -> str:
    return f"ui_{block_id}_{field}"


def init_ui_state() -> None:
    for block_id, config in UI_BLOCKS.items():
        st.session_state.setdefault(ui_key(block_id, "visible"), True)
        st.session_state.setdefault(ui_key(block_id, "order"), int(config["order"]))
        st.session_state.setdefault(ui_key(block_id, "title_size"), int(config["title_size"]))
        st.session_state.setdefault(ui_key(block_id, "body_size"), int(config["body_size"]))
        st.session_state.setdefault(ui_key(block_id, "width"), config["width"])


def ui_visible(block_id: str) -> bool:
    return bool(st.session_state.get(ui_key(block_id, "visible"), True))


def ui_order(block_id: str) -> int:
    return int(st.session_state.get(ui_key(block_id, "order"), UI_BLOCKS[block_id]["order"]))


def ui_width_style(block_id: str) -> str:
    width = st.session_state.get(ui_key(block_id, "width"), "wide")
    if width == "compact":
        return "width: calc(100% - 160px); max-width: 860px; margin-left: 80px; margin-right: auto;"
    if width == "full":
        return "width: calc(100% - 48px); max-width: none; margin-left: 24px; margin-right: 24px;"
    return "width: calc(100% - 96px); max-width: 1160px; margin-left: 48px; margin-right: auto;"


def ui_block_label(block_id: str) -> None:
    if not st.session_state.get("ui_edit_mode"):
        return
    label = UI_BLOCKS.get(block_id, {}).get("label", block_id)
    order = ui_order(block_id)
    st.caption(f"组件：{block_id} | {label} | 顺序 {order}")


def ui_text_style(block_id: str, kind: str = "body") -> str:
    field = "title_size" if kind == "title" else "body_size"
    size = int(st.session_state.get(ui_key(block_id, field), UI_BLOCKS[block_id][field]))
    weight = 760 if kind == "title" else 400
    return f"font-size:{size}px; font-weight:{weight};"


def current_scope_option(registry: dict, metas: list[dict], key: str) -> dict:
    options = document_options(registry, metas)
    labels = [option["label"] for option in options]
    selected_label = st.session_state.get(key)
    if selected_label not in labels:
        selected_label = labels[0]
    return next(option for option in options if option["label"] == selected_label)


def render_ui_editor_controls() -> None:
    st.session_state.setdefault("ui_edit_mode", False)
    edit_mode = st.toggle("界面编辑模式", value=st.session_state.ui_edit_mode, key="ui_edit_mode_toggle")
    st.session_state.ui_edit_mode = edit_mode
    if not edit_mode:
        return
    st.caption("只控制界面块，不影响 RAG 检索和 IFC 解析。顺序值越小越靠前。")
    if st.button("恢复默认界面配置", use_container_width=True):
        for block_id, config in UI_BLOCKS.items():
            st.session_state[ui_key(block_id, "visible")] = True
            st.session_state[ui_key(block_id, "order")] = int(config["order"])
            st.session_state[ui_key(block_id, "title_size")] = int(config["title_size"])
            st.session_state[ui_key(block_id, "body_size")] = int(config["body_size"])
            st.session_state[ui_key(block_id, "width")] = config["width"]
        st.rerun()
    for block_id, config in UI_BLOCKS.items():
        with st.expander(f"{block_id} | {config['label']}", expanded=False):
            st.checkbox("显示这个组件", key=ui_key(block_id, "visible"))
            st.number_input("顺序", min_value=1, max_value=999, step=5, key=ui_key(block_id, "order"))
            st.slider("标题字号", min_value=11, max_value=28, key=ui_key(block_id, "title_size"))
            st.slider("正文字号", min_value=10, max_value=22, key=ui_key(block_id, "body_size"))
            st.selectbox("宽度", ["wide", "compact", "full"], key=ui_key(block_id, "width"))


def render_sidebar(registry: dict, vector_config: dict) -> tuple[bool, bool, int]:
    standards = registry.get("standards", {})
    updated_at = registry.get("updated_at", "未记录")
    if "active_module" not in st.session_state:
        st.session_state.active_module = "规范问答"

    def nav_button(label: str, key: str) -> None:
        active = st.session_state.active_module == label
        if st.button(label, key=key, type="primary" if active else "secondary", use_container_width=True):
            st.session_state.active_module = label
            st.rerun()

    with st.sidebar:
        st.markdown(
            """
<div class="sidebar-title">建筑消防审查</div>
<div class="sidebar-version">规范 RAG 与 BIM 辅助审查工作台</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-section-label">核心功能</div>', unsafe_allow_html=True)
        nav_button("规范问答", "nav_qa")
        nav_button("BIM 模型审查", "nav_bim")

        st.markdown('<div class="sidebar-section-label">支撑模块</div>', unsafe_allow_html=True)
        nav_button("规范库", "nav_library")
        nav_button("查询历史", "nav_history")
        nav_button("系统状态", "nav_status")

        st.divider()
        render_ui_editor_controls()
        st.divider()
        st.button("上传新规范", use_container_width=True)
        if ui_visible("sidebar_library"):
            ui_block_label("sidebar_library")
            st.markdown('<div class="sidebar-section-label">规范库状态</div>', unsafe_allow_html=True)
            for standard_id, item in sorted(
                standards.items(),
                key=lambda pair: (-int(pair[1].get("priority", 0)), pair[0]),
            ):
                title = item.get("title", STANDARD_NAMES.get(standard_id, standard_id))
                source_file = item.get("file") or f"{standard_id}.pdf"
                st.markdown(
                    f'<div class="sidebar-doc">{escape(title)}<br><strong>{escape(standard_id)}</strong><br><span>{escape(source_file)}</span></div>',
                    unsafe_allow_html=True,
                )

            chunks = vector_config.get("chunks", "未知")
            model = vector_config.get("embedding_model", "未知")
            st.markdown(
                f"""
<div class="sidebar-kv">
  <div><strong>当前适用范围：</strong><span style="float:right;">全库</span></div>
  <div><strong>数据更新时间：</strong><span style="float:right;">{escape(str(updated_at))}</span></div>
  <div><strong>规范版本状态：</strong><span style="float:right;color:#86efac;">现行有效</span></div>
  <div><strong>知识片段：</strong><span style="float:right;">{escape(str(chunks))}</span></div>
  <div><strong>向量模型：</strong><span style="float:right;">{escape(str(model))}</span></div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.markdown(
                """
<div class="sidebar-alert">
  包含新旧规范关联提醒。系统会自动提示通用规范、专项规范和旧版设计规范之间的复核风险。
</div>
""",
                unsafe_allow_html=True,
            )

        expert_mode = False
        top_k = 5
        use_api_rerank = False
        if ui_visible("sidebar_settings"):
            st.divider()
            ui_block_label("sidebar_settings")
            st.markdown('<div class="sidebar-section-label">高级设置</div>', unsafe_allow_html=True)
            expert_mode = st.toggle("Expert Mode", value=False)
            if st.button("清空对话与上下文", use_container_width=True):
                st.session_state.pop("conversation_state", None)
                st.session_state.pop("conversation_history", None)
                st.session_state.pop("chat_turns", None)
                st.session_state.pop("pending_prompt", None)
                st.session_state.pop("ifc_review_turns", None)
                st.session_state.pop("ifc_model_answers", None)
                st.rerun()
            top_k = st.slider("引用依据数量", min_value=3, max_value=10, value=5)
            if expert_mode:
                use_api_rerank = st.toggle(
                    "二阶段语义重排",
                    value=False,
                    help="调用独立 reranker 模型重新排序候选证据，不负责生成回答。默认关闭，用于 A/B 测试召回质量和成本。",
                )

        history = st.session_state.get("conversation_history") or []
        if history:
            st.divider()
            st.markdown("**历史问题列表**")
            for item in history[-5:]:
                st.caption(f"{item['question']} | {item['top_standard']} {item['top_article'] or ''}")

    return expert_mode, use_api_rerank, top_k


def render_scope_selector(registry: dict, metas: list[dict], key: str = "selected_scope_label", block_id: str | None = None) -> dict:
    options = document_options(registry, metas)
    labels = [option["label"] for option in options]
    selected_label = st.session_state.get(key)
    if selected_label not in labels:
        selected_label = labels[0]

    with st.container(border=True):
        if block_id:
            ui_block_label(block_id)
        title_style = ui_text_style(block_id, "title") if block_id else ""
        body_style = ui_text_style(block_id, "body") if block_id else ""
        st.markdown(f'<div class="scope-card-title" style="{title_style}">知识库使用范围</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="scope-card-caption" style="{body_style}">默认自动识别。需要限定某一份规范时，从当前知识库文件中选择。</div>',
            unsafe_allow_html=True,
        )
        selected_label = st.selectbox(
            "选择检索范围",
            labels,
            index=labels.index(selected_label),
            label_visibility="collapsed",
            key=key,
        )
        selected = next(option for option in options if option["label"] == selected_label)
        st.caption(selected["caption"])
    return selected


def render_mode_nav(active_module: str) -> str:
    if not ui_visible("top_bar"):
        return active_module
    ui_block_label("top_bar")
    st.markdown(
        f"""
<div class="mode-nav" style="{ui_width_style('top_bar')}">
  <div>
    <div class="mode-nav-title" style="{ui_text_style('top_bar', 'title')}">{escape(active_module)}</div>
    <div class="mode-nav-caption" style="{ui_text_style('top_bar', 'body')}">按左侧模块切换工作区；当前页面只展示一个完整任务流程。</div>
  </div>
  <div class="top-actions">
    <div class="top-search">搜索规范、条文或项目问题...</div>
    <div class="top-chip">专家模式</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    return active_module


@st.cache_data(show_spinner=False)
def parse_uploaded_ifc(file_name: str, content: bytes) -> dict:
    IFC_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", file_name or "model.ifc")
    digest = hashlib.sha256(content).hexdigest()[:12]
    path = IFC_UPLOAD_DIR / f"{digest}_{safe_name}"
    path.write_bytes(content)
    return parse_ifc_file(path)


def normalize_ifc_review_query(query: str) -> str:
    if "基于IFC解析" not in (query or ""):
        return query
    context_match = re.search(r"项目条件补充[:：]\s*(.+)$", query, flags=re.S)
    project_context = clean_text(context_match.group(1), 180) if context_match else ""
    context_suffix = f" 项目条件补充 {project_context}" if project_context else ""
    text = re.sub(r"（对象名：[^）]*）", "", query)
    text = re.sub(r"项目条件补充[:：].+$", "", text, flags=re.S)
    text = re.sub(r"\b\d{5,}\b", "", text)
    text = re.sub(r"\b\d+(?:\.\d+)?mm\s*x\s*\d+(?:\.\d+)?mm\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()

    width_match = re.search(r"宽度约\s*(\d+(?:\.\d+)?)\s*m", text)
    height_match = re.search(r"建筑高度约\s*(\d+(?:\.\d+)?)\s*m", text)

    if "门" in text and width_match:
        width = width_match.group(1)
        if "疑似疏散出口门" in text:
            return (
                f"IFC模型审查 门宽 {width}m 疑似疏散出口门 疏散门 净宽 最低要求 "
                "建筑防火通用规范 GB 55037 疏散出口门 净宽度 不应小于 是否满足 人工确认建筑类型和门部位"
                + context_suffix
            )
        return (
            f"IFC模型审查 门宽 {width}m 用途未识别 普通门 疏散出口门 疏散门 住宅户门 "
            "最低净宽要求 建筑防火通用规范 GB 55037 判断是否可能满足 并说明必须人工确认建筑类型、门部位、是否作为疏散出口"
            + context_suffix
        )

    if height_match:
        height = height_match.group(1)
        return (
            f"IFC模型审查 建筑高度 {height}m 楼层数 建筑分类 高层建筑 民用建筑 "
            "建筑防火通用规范 建筑设计防火规范 需要优先核对哪些消防审查要求"
            + context_suffix
        )

    return text + context_suffix


def enrich_ifc_question(question: str, project_context: str) -> str:
    project_context = clean_text(project_context, 240)
    if not project_context:
        return question
    return f"{question}\n项目条件补充：{project_context}"


def ifc_model_context(summary: dict, project_context: str = "") -> str:
    counts = summary.get("counts") or {}
    building_use = summary.get("building_use") or {}
    height = summary.get("building_height_m")
    parts = [
        f"基于IFC解析：模型楼层数 {counts.get('IfcBuildingStorey', 0)}，空间数 {counts.get('IfcSpace', 0)}，门数量 {counts.get('IfcDoor', 0)}，楼梯数量 {counts.get('IfcStair', 0)}。",
        f"推算建筑高度：{height}m。" if height is not None else "推算建筑高度：未识别。",
        f"建筑使用功能：{building_use.get('value', '未稳定识别')}，识别置信度：{building_use.get('confidence', '低')}。",
    ]
    if project_context:
        parts.append(f"项目条件补充：{clean_text(project_context, 240)}。")
    return " ".join(parts)


def answer_ifc_model_question(summary: dict, question: str, project_context: str = "") -> dict:
    q = clean_text(question)
    counts = summary.get("counts") or {}
    height = summary.get("building_height_m")
    storey_analysis = summary.get("storey_analysis") or {}
    building_use = summary.get("building_use") or {}
    doors = summary.get("doors") or []
    spaces = summary.get("spaces") or []

    if ("平均" in q and ("层" in q or "楼层" in q) and "高" in q) or re.search(r"层高|每层.*高", q):
        average = storey_analysis.get("average_interval_m")
        intervals = storey_analysis.get("intervals") or []
        if average is not None:
            detail = "；".join([f"{item['from']}->{item['to']}：{item['height_m']}m" for item in intervals[:8]])
            return {
                "title": "IFC 模型内容查询",
                "answer": f"按 IFC 楼层高程相邻差值计算，平均楼层高约 {average}m。",
                "basis": f"模型推算建筑高度约 {height}m；楼层高程间距：{detail or '未列出'}。这是模型数据计算结果，不是规范合规结论。",
            }
        if height is not None and counts.get("IfcBuildingStorey", 0) > 1:
            avg = round(float(height) / (int(counts["IfcBuildingStorey"]) - 1), 3)
            return {
                "title": "IFC 模型内容查询",
                "answer": f"按建筑高度跨度 / 相邻楼层段数粗算，平均楼层高约 {avg}m。",
                "basis": f"模型推算高度 {height}m，楼层数 {counts.get('IfcBuildingStorey')}。IFC 未提供完整楼层间距明细，结果需复核。",
            }

    if re.search(r"建筑.*多高|建筑高度|总高|高度", q):
        return {
            "title": "IFC 模型内容查询",
            "answer": f"IFC 楼层高程推算建筑高度约 {height}m。" if height is not None else "当前 IFC 未稳定推算出建筑高度。",
            "basis": "该高度来自 IfcBuildingStorey 的最高与最低 Elevation 差值；不是规范定义下所有场景的正式建筑高度。",
        }

    if re.search(r"几层|多少层|楼层", q):
        levels = storey_analysis.get("levels") or []
        level_text = "；".join([f"{item['name']}({item['elevation_m']}m)" for item in levels[:12]])
        return {
            "title": "IFC 模型内容查询",
            "answer": f"模型中识别到 {counts.get('IfcBuildingStorey', 0)} 个 IfcBuildingStorey。",
            "basis": f"楼层高程：{level_text or '未读取到楼层高程'}。",
        }

    if re.search(r"门|门宽|宽度", q):
        with_width = [door for door in doors if door.get("width_m") is not None]
        widths = sorted({round(float(door["width_m"]), 3) for door in with_width})
        sample = "；".join(
            [
                f"{door.get('storey') or '未识别楼层'} {clean_text(door.get('name'), 38)}：{round(float(door['width_m']), 3)}m"
                for door in with_width[:8]
            ]
        )
        return {
            "title": "IFC 模型内容查询",
            "answer": f"模型中识别到 {counts.get('IfcDoor', 0)} 扇门，其中 {len(with_width)} 扇读取到宽度字段；宽度类型包括 {', '.join([str(item) + 'm' for item in widths]) or '未识别'}。",
            "basis": f"样例：{sample or '无可展示门宽'}。门是否为疏散门/户门/安全出口门，需要空间拓扑和人工确认。",
        }

    if re.search(r"楼梯|梯段", q):
        return {
            "title": "IFC 模型内容查询",
            "answer": f"模型中识别到 {counts.get('IfcStair', 0)} 个 IfcStair 楼梯构件。",
            "basis": "当前仅统计 IFC 构件数量；楼梯是否作为疏散楼梯、净宽是否满足规范，需要进一步读取几何宽度和连接关系。",
        }

    if re.search(r"空间|房间|房间数", q):
        sample = "；".join([clean_text(space.get("name"), 32) for space in spaces[:12]])
        return {
            "title": "IFC 模型内容查询",
            "answer": f"模型中识别到 {counts.get('IfcSpace', 0)} 个空间。",
            "basis": f"空间样例：{sample or '未读取到空间名称'}。",
        }

    if re.search(r"建筑类型|使用功能|用途|住宅|公共建筑", q):
        extra = f"；用户补充条件：{clean_text(project_context, 160)}" if project_context else ""
        return {
            "title": "IFC 模型内容查询",
            "answer": f"系统从 IFC 命名和空间信息中推断为：{building_use.get('value', '未稳定识别')}，置信度 {building_use.get('confidence', '低')}。",
            "basis": f"{building_use.get('reason', 'IFC 未提供可直接确定建筑功能的字段')}{extra}。最终建筑类型应以项目资料或人工输入为准。",
        }

    return {
        "title": "IFC 模型内容查询",
        "answer": "这个问题更像需要进一步指定查询对象。当前可查询：楼层数量、建筑高度、平均层高、空间数量、门宽、楼梯数量、建筑使用功能推断。",
        "basis": "如果问题涉及“是否合规、规范要求、疏散、防火、消防设施”，请放到消防合规审查中处理。",
    }


def render_ifc_model_answer(item: dict) -> None:
    with st.container(border=True):
        st.markdown(f"**{escape(item.get('question', '模型查询'))}**")
        st.markdown(f"**{escape(item.get('answer', {}).get('answer', '未生成回答'))}**")
        basis = item.get("answer", {}).get("basis")
        if basis:
            st.caption(basis)


def build_isolated_review_turn(
    query: str,
    *,
    scope: dict,
    top_k: int,
    use_api_rerank: bool,
    registry: dict,
    abolished: dict,
    vector_config: dict,
) -> dict:
    rewritten = normalize_ifc_review_query(query)
    forced_standard = scope.get("standard_id") or ""
    routed_query = f"{forced_standard} {rewritten}" if forced_standard else rewritten
    route = infer_route(routed_query, registry)
    results = hybrid_search(
        routed_query,
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        use_api_rerank=use_api_rerank,
        rerank_top_n=top_k,
        final_top_k=top_k,
    )
    confidence = confidence_from_results(results, route.primary + route.secondary, abolished, routed_query)
    llm_answer = None
    llm_error = None
    if os.getenv("DEEPSEEK_API_KEY"):
        try:
            llm_answer = generate_answer(
                query,
                results[:5],
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                retrieval_question=rewritten,
                is_followup=False,
                is_reference_question=False,
            )
        except (LLMAnswerError, ValueError, KeyError, IndexError) as exc:
            llm_error = str(exc)[:500]
    return {
        "question": query,
        "rewritten": rewritten,
        "rewrite_reasons": ["IFC 项目审查问题在独立模块内检索，不写入连续问答上下文。"],
        "scope": scope.get("label", "自动识别"),
        "scope_standard_id": forced_standard,
        "route": {"primary": route.primary, "secondary": route.secondary, "reasons": route.reasons},
        "confidence": confidence,
        "results": results,
        "contextual_followup": False,
        "context_decision": {
            "mode": "new_topic",
            "use_previous_context": False,
            "resolved_question": rewritten,
            "reason": "IFC 项目审查独立运行，不继承聊天上下文。",
            "context_used": [],
            "source": "ifc_isolated",
        },
        "context_evidence_count": 0,
        "llm_answer": llm_answer,
        "llm_error": llm_error,
        "vector_model": vector_config.get("embedding_model"),
        "expert_mode": False,
    }


def render_ifc_panel(*, scope: dict, top_k: int, use_api_rerank: bool, registry: dict, abolished: dict, vector_config: dict, expert_mode: bool) -> None:
    with st.container():
        st.markdown('<div class="workbench-title">模型事实与消防合规审查</div>', unsafe_allow_html=True)
        st.caption("先解析 BIM 模型事实，再结合自然语言项目条件进入规范审查；模型内容查询和合规判断分开处理。")
        uploaded = st.file_uploader(
            "上传 IFC 模型",
            type=["ifc"],
            key="ifc_model_upload",
            help="支持标准 IFC 文件。当前阶段先做字段抽取和审查问题生成，不做完整几何路径计算。",
        )
        if not uploaded:
            st.caption("处理链路：IFC 解析 -> 字段归一 -> 场景识别 -> 生成审查问题 -> RAG 检索规范 -> 合规判断。")
            return

        try:
            summary = parse_uploaded_ifc(uploaded.name, uploaded.getvalue())
        except Exception as exc:
            st.error(f"IFC 解析失败：{str(exc)[:500]}")
            return

        project = summary.get("project") or {}
        counts = summary.get("counts") or {}
        scenes = summary.get("detected_scenes") or {}
        building_use = summary.get("building_use") or {}
        field_quality = summary.get("field_quality") or {}
        metric_cols = st.columns(5)
        metric_cols[0].metric("楼层", counts.get("IfcBuildingStorey", 0))
        metric_cols[1].metric("空间", counts.get("IfcSpace", 0))
        metric_cols[2].metric("门", counts.get("IfcDoor", 0))
        metric_cols[3].metric("楼梯", counts.get("IfcStair", 0))
        height = summary.get("building_height_m")
        metric_cols[4].metric("推算高度", f"{height}m" if height is not None else "未识别")

        st.caption(
            f"文件：{summary.get('source_file', uploaded.name)}；"
            f"Schema：{project.get('schema') or '未识别'}；"
            f"项目：{project.get('project_name') or '未命名'}；"
            f"建筑：{project.get('building_name') or '未命名'}"
        )
        st.warning(
            f"建筑使用功能：{building_use.get('value', '未稳定识别')}；"
            f"识别置信度：{building_use.get('confidence', '低')}。{building_use.get('reason', '')}"
        )
        if field_quality.get("missing"):
            st.caption("当前缺失字段：" + "、".join(field_quality.get("missing", [])))
        if field_quality.get("available"):
            st.caption("可用字段：" + "、".join(field_quality.get("available", [])))

        with st.container(border=True):
            st.markdown("**项目条件补充**")
            st.caption("这里用自然语言填写，不做下拉选择。系统会把这些条件带入 IFC 项目审查检索。")
            project_context = st.text_area(
                "建筑类型、用途、项目条件",
                key="ifc_project_context",
                height=84,
                placeholder="例如：多层住宅，地上4层，无地下室；当前关注户门、疏散门和楼梯净宽。也可以写：商业公共建筑、地下汽车库、是否设置自动灭火系统等。",
                label_visibility="collapsed",
            )

        tabs = st.tabs(["模型内容查询", "消防合规审查", "解析字段", "处理链路"])
        with tabs[0]:
            st.markdown("**查询 IFC 模型内容**")
            st.caption("这里只回答模型里已经解析出的事实，例如楼层数、平均层高、门宽、空间数量，不检索规范条文。")
            model_question = st.text_area(
                "模型内容问题",
                key="ifc_model_question",
                height=86,
                placeholder="例如：这栋建筑平均楼层多高？模型里有几扇门？Level 1 有哪些门宽？这栋建筑有几层？",
                label_visibility="collapsed",
            )
            if st.button("查询模型内容", key="ifc_model_query_btn", use_container_width=True):
                if clean_text(model_question):
                    answer = answer_ifc_model_question(summary, model_question, project_context)
                    st.session_state.setdefault("ifc_model_answers", []).append({"question": model_question, "answer": answer})
                    st.rerun()
                else:
                    st.warning("请先输入要查询的模型内容。")
            if st.session_state.get("ifc_model_answers"):
                st.divider()
                st.markdown("**模型查询结果**")
                for item in st.session_state.get("ifc_model_answers", [])[-5:]:
                    render_ifc_model_answer(item)

        with tabs[1]:
            st.markdown("**消防合规审查**")
            st.caption("这里会把 IFC 解析摘要和项目条件带入规范 RAG，用于判断是否满足消防规范。")
            custom_question = st.text_area(
                "合规审查问题",
                key="ifc_custom_question",
                height=96,
                placeholder="例如：Level 1 的 0.762m 门如果是住宅户门是否满足要求？这栋多层住宅的疏散楼梯数量需要怎样复核？",
                label_visibility="collapsed",
            )
            if st.button("检索这个合规问题", key="ifc_custom_review", use_container_width=True):
                if clean_text(custom_question):
                    full_question = f"{ifc_model_context(summary, project_context)} 审查问题：{clean_text(custom_question, 500)}"
                    turn = build_isolated_review_turn(
                        full_question,
                        scope=scope,
                        top_k=top_k,
                        use_api_rerank=use_api_rerank,
                        registry=registry,
                        abolished=abolished,
                        vector_config=vector_config,
                    )
                    st.session_state.setdefault("ifc_review_turns", []).append(turn)
                    st.rerun()
                else:
                    st.warning("请先输入要审查的问题。")

            questions = build_ifc_review_questions(summary)
            with st.expander("查看系统建议审查问题", expanded=False):
                for idx, item in enumerate(questions):
                    with st.container(border=True):
                        st.markdown(f"**{item['title']}**")
                        st.write(item["question"])
                        if st.button("检索这条建议", key=f"ifc_question_{idx}", use_container_width=True):
                            review_question = enrich_ifc_question(item["question"], project_context)
                            turn = build_isolated_review_turn(
                                review_question,
                                scope=scope,
                                top_k=top_k,
                                use_api_rerank=use_api_rerank,
                                registry=registry,
                                abolished=abolished,
                                vector_config=vector_config,
                            )
                            st.session_state.setdefault("ifc_review_turns", []).append(turn)
                            st.rerun()

            if st.session_state.get("ifc_review_turns"):
                st.divider()
                st.markdown("**合规审查结果**")
                for turn in st.session_state.get("ifc_review_turns", [])[-5:]:
                    render_chat_turn(turn, abolished, expert_mode=expert_mode)

        with tabs[2]:
            field_cols = st.columns(3)
            field_cols[0].metric("疑似汽车库空间", scenes.get("garage_spaces", 0))
            field_cols[1].metric("疑似中庭空间", scenes.get("atrium_spaces", 0))
            field_cols[2].metric("带宽度字段的门", scenes.get("doors_with_width", 0))

            doors = summary.get("doors") or []
            if doors:
                st.markdown("**门与疏散出口候选**")
                door_rows = [
                    {
                        "名称": row.get("name"),
                        "楼层": row.get("storey") or "未识别",
                        "宽度(m)": row.get("width_m"),
                        "防火性能": row.get("fire_rating") or "未识别",
                        "疑似出口": "是" if row.get("is_exit_like") else "否",
                    }
                    for row in doors[:20]
                ]
                st.dataframe(door_rows, use_container_width=True, hide_index=True)

            fire_elements = summary.get("fire_related_elements") or []
            if fire_elements:
                st.markdown("**消防/疏散相关构件候选**")
                st.dataframe(fire_elements[:20], use_container_width=True, hide_index=True)

            for limit in summary.get("parser_limits", []):
                st.caption(f"- {limit}")

        with tabs[3]:
            for item in ifc_processing_chain():
                st.markdown(f"**{item['stage']}**")
                st.caption(item["detail"])


def confidence_from_results(results: list[dict], route_primary: list[str], abolished: dict, question: str = "") -> dict:
    intent = detect_query_intent(question)
    if intent.asks_version_compare:
        return {
            "level": "low",
            "label": "低",
            "can_answer": False,
            "reason": "这是版本差异/修订对比问题，需要同时收录待比较版本并建立修订映射；当前普通规范检索不能可靠回答。",
        }
    if intent.asks_local_rule:
        return {
            "level": "low",
            "label": "低",
            "can_answer": False,
            "reason": "问题涉及地方标准或地方审查口径，但当前知识库只收录已入库的国家规范，不能把国标片段扩展成地方结论。",
        }
    if intent.lacks_project_context:
        return {
            "level": "low",
            "label": "低",
            "can_answer": False,
            "reason": "问题缺少建筑类型、部位、面积/距离/宽度等关键项目条件，不能直接判断是否合规。",
        }
    if intent.page_refs:
        return {
            "level": "low",
            "label": "低",
            "can_answer": False,
            "reason": "这是页码定位或原文查看请求，系统可以展示候选页片段，但不应把整页内容改写成确定结论。",
        }

    if not results:
        return {
            "level": "low",
            "label": "低",
            "can_answer": False,
            "reason": "没有召回到可用于审查的规范片段。",
        }

    top = results[0]
    second = results[1] if len(results) > 1 else None
    top_score = float(top.get("hybrid_score") or 0)
    second_score = float(second.get("hybrid_score") or 0) if second else 0
    margin = top_score - second_score
    has_article = bool(top.get("article_no"))
    has_page = top.get("page") is not None
    has_text = bool(clean_text(top.get("text", "")))
    route_match = not route_primary or top.get("standard_id") in route_primary
    deprecated = bool(is_abolished(top, abolished))
    exclusion_conflict = exclusion_conflict_reason(question, top.get("text", ""))

    if exclusion_conflict:
        return {
            "level": "low",
            "label": "低",
            "can_answer": False,
            "reason": exclusion_conflict,
        }

    if has_article and has_page and has_text and route_match and deprecated:
        return {
            "level": "medium",
            "label": "中",
            "can_answer": True,
            "reason": "已召回到明确旧版/专项规范条文，但该条文存在废止或替代提醒，最终结论需要结合现行通用规范复核。",
        }

    if has_article and has_page and has_text and route_match and not deprecated and margin >= 3:
        return {
            "level": "high",
            "label": "高",
            "can_answer": True,
            "reason": "已召回到明确标准、条文号、页码和原文片段，且首条证据排序优势明显。",
        }
    if has_page and has_text and route_match and not deprecated:
        return {
            "level": "medium",
            "label": "中",
            "can_answer": True,
            "reason": "已召回到相关原文，但条文号、排序优势或字段完整性仍需人工复核。",
        }
    return {
        "level": "low",
        "label": "低",
        "can_answer": False,
        "reason": "召回证据不足、适用规范不明确，或首条结果存在废止/替代风险。",
    }


def field_recall_status(item: dict | None) -> list[tuple[str, bool, str]]:
    return evidence_field_recall_status(item)


def evidence_sentence(question: str, text: str) -> str:
    compact = clean_text(text)
    if not compact:
        return ""

    sentences = re.split(r"(?<=[。；;])", compact)
    query_chars = {ch for ch in question if "\u4e00" <= ch <= "\u9fff"}

    def score(sentence: str) -> int:
        overlap = sum(1 for ch in query_chars if ch in sentence)
        number_bonus = 8 if re.search(r"\d+(?:\.\d+)?\s*(?:m|h|㎡|%|人|层)", sentence, re.IGNORECASE) else 0
        article_bonus = 4 if re.search(r"\d+\.\d+(?:\.\d+)?", sentence) else 0
        return overlap + number_bonus + article_bonus

    best = max(sentences or [compact], key=score)
    return clean_text(best, 260)


def answer_block(question: str, results: list[dict], confidence: dict, abolished: dict) -> None:
    top = results[0] if results else None
    level_class = confidence_class(confidence["level"])

    if not confidence["can_answer"] or not top:
        st.markdown(
            f"""
<div class="section-card summary-card">
  <div class="workbench-title">规范解读摘要 <span class="pill">证据置信度：{confidence['label']}</span></div>
  <p><strong>当前检索结果不足以生成可靠答案。</strong></p>
  <p>系统不会根据模型常识补写结论；请扩大检索范围、改写问题，或人工查看下方候选原文。</p>
  <p class="{level_class}">原因：{escape(confidence['reason'])}</p>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    sentence = evidence_sentence(question, top.get("text", ""))
    summary = sentence or "已召回到相关条文，但未自动抽取出稳定摘要，请展开原文依据后人工复核。"
    st.markdown(
        f"""
<div class="section-card summary-card">
  <div class="workbench-title">规范解读摘要 <span class="pill">证据置信度：{confidence['label']}</span></div>
  <p>根据当前召回结果，系统已找到可审查依据：</p>
  <ul>
    <li><strong>命中规范：</strong>{escape(standard_label(top['standard_id']))}</li>
    <li><strong>条文位置：</strong>第 {escape(str(top.get('article_no') or '未识别'))} 条，{escape(page_label(top.get('page')))}</li>
    <li><strong>摘要：</strong>{escape(summary)}</li>
  </ul>
</div>
""",
        unsafe_allow_html=True,
    )

    if is_abolished(top, abolished):
        st.markdown(
            """
<div class="section-card warning-card">
  <strong>需人工复核</strong>
  <p>首条证据存在废止或替代风险，不能直接作为最终答案。</p>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
<div class="section-card warning-card">
  <strong>需人工复核</strong>
  <p>本条为基于召回原文生成的审查摘要。若问题涉及具体项目条件、地方标准或新旧规范冲突，应以展开的原文片段和人工核验为准。</p>
</div>
""",
            unsafe_allow_html=True,
        )


def llm_answer_block(answer: dict | None, error: str | None) -> None:
    st.markdown("### 大语言判断")
    if error:
        st.error("大语言模型没有返回可靠结果，系统不会补写答案。")
        st.caption(error)
        return
    if not answer:
        st.info("未启用大语言生成。当前仅展示检索证据与证据置信度。")
        return

    certainty = answer.get("certainty", "low")
    color = {"high": "green", "medium": "orange", "low": "red"}.get(certainty, "red")
    label = {"high": "高", "medium": "中", "low": "低"}.get(certainty, "低")
    st.badge(f"模型判断置信度：{label}", color=color)
    st.write("是否可回答：" + ("是" if answer.get("can_answer") else "否"))
    st.write(answer.get("conclusion", "模型未给出结论。"))
    basis = answer.get("basis")
    if basis:
        st.caption(f"判断依据：{basis}")
    citations = answer.get("citations") or []
    if citations:
        st.caption("引用证据编号：" + "、".join([str(item) for item in citations]))
    missing_fields = answer.get("missing_fields") or []
    if missing_fields:
        st.warning("缺失字段：" + "、".join([str(item) for item in missing_fields]))
    note = answer.get("note")
    if note:
        if certainty != "high":
            st.warning(note)
        else:
            st.caption(note)


def field_status_block(top: dict | None) -> None:
    st.markdown("### 字段召回检查")
    cols = st.columns(5)
    for col, (name, ok, value) in zip(cols, field_recall_status(top)):
        col.metric(name, "已召回" if ok else "未召回", value)


def citation_card(idx: int, item: dict, abolished: dict, *, expert_mode: bool) -> None:
    abolished_info = is_abolished(item, abolished)
    title = f"{idx}. {standard_label(item['standard_id'])}"
    article = item.get("article_no") or "未识别条文号"
    page = item.get("page")
    text = clean_text(item.get("text", ""))
    notes = item.get("policy_notes") or []

    with st.container(border=True):
        top_line = st.columns([5, 1.1])
        top_line[0].markdown(f"**{title}**")
        if abolished_info:
            top_line[1].markdown('<span class="score-low">相关度：需复核</span>', unsafe_allow_html=True)
        else:
            top_line[1].markdown('<span class="score-high">相关度：高</span>', unsafe_allow_html=True)

        st.write(f"条文号：`{article}`　页码：`{page_label(page)}`")
        if item.get("table_no"):
            st.write(f"表号：`表 {item.get('table_no')}`")
        if notes:
            st.caption("命中原因：" + "；".join(notes[:3]))

        short = evidence_sentence("", text) or clean_text(text, 180)
        st.write(short)

        with st.expander("查看原文片段和页码定位", expanded=idx == 1):
            if abolished_info:
                st.warning(
                    f"废止/替代提醒：{abolished_info['source_standard_id']} {abolished_info['source_article']} "
                    f"需核对 {abolished_info['replaced_by_standard_id']}。"
                )
            escaped_text = escape(text[:1800])
            st.markdown(
                f"""
<div style="font-size:14px;line-height:1.75;background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:12px;color:#0f172a;white-space:pre-wrap;">
{escaped_text}
</div>
""",
                unsafe_allow_html=True,
            )

        if expert_mode:
            with st.expander("专家审查：排序与召回细节", expanded=False):
                st.json(
                    {
                        "hybrid_score": item.get("hybrid_score"),
                        "bm25_score": item.get("bm25_score"),
                        "vector_score": item.get("vector_score"),
                        "rrf_score": item.get("fused_score"),
                        "api_rerank_score": item.get("api_rerank_score"),
                        "table_no": item.get("table_no"),
                        "table_score": item.get("table_score"),
                        "chunk_id": item.get("chunk_id"),
                        "row_id": item.get("row_id"),
                    }
                )


def source_brief(item: dict) -> str:
    source_file = item.get("source_file") or item.get("standard_id") or "未知文件"
    article = item.get("article_no") or "未识别"
    page = item.get("page")
    return f"{source_file} | {item.get('standard_id')} | 第 {article} 条 | PDF物理页第 {page} 页"


def page_label(page: object) -> str:
    return f"PDF物理页第 {page} 页（含封面、目录）" if page is not None else "PDF物理页未识别"


def source_brief_display(item: dict) -> str:
    source_file = item.get("source_file") or item.get("standard_id") or "未知文件"
    standard_id = item.get("standard_id") or "未知标准"
    article = item.get("article_no") or "未识别"
    page = item.get("page")
    page_text = page_label(page)
    return f"来源文件：{source_file}；标准号：{standard_id}；条文：第 {article} 条；页码：{page_text}"


def highlight_answer_html(text: str) -> str:
    html = escape(text or "")
    scenario_terms = [
        "商业综合体",
        "商业建筑",
        "中庭",
        "周围连通空间",
        "高层公共建筑",
        "高层医疗建筑",
        "住宅建筑",
        "汽车库",
        "地下汽车库",
        "防火分区",
        "防火卷帘",
        "疏散楼梯",
        "疏散出口",
        "疏散门",
        "疏散走道",
    ]
    scenario_hits = 0
    for term in sorted(scenario_terms, key=len, reverse=True):
        if scenario_hits >= 3:
            break
        count = min(3 - scenario_hits, html.count(escape(term)))
        if count:
            html = html.replace(escape(term), f'<span class="hl-scenario">{escape(term)}</span>', count)
            scenario_hits += count
    html = re.sub(r"(GB\s*\d{5}-\d{4})", r'<span class="hl-source">\1</span>', html, count=3)
    html = re.sub(r"(第\s*\d+(?:\.\d+[A-Za-z]?)*\s*条)", r'<span class="hl-source">\1</span>', html, count=3)
    html = re.sub(r"(PDF物理页第\s*\d+\s*页(?:（含封面、目录）)?)", r'<span class="hl-page">\1</span>', html, count=2)
    html = re.sub(
        r"(\d+(?:\.\d+)?\s*(?:m|h|㎡|m2|人|层|%))",
        r'<span class="hl-number">\1</span>',
        html,
        count=5,
        flags=re.IGNORECASE,
    )
    return html


def final_answer_text(question: str, results: list[dict], confidence: dict, llm_answer: dict | None, llm_error: str | None) -> tuple[str, str, str]:
    evidence_can_answer = bool(confidence.get("can_answer"))
    llm_can_answer = bool(llm_answer and llm_answer.get("can_answer"))
    if evidence_can_answer and llm_can_answer and llm_answer and llm_answer.get("conclusion"):
        conclusion = str(llm_answer.get("conclusion"))
        basis = str(llm_answer.get("basis") or "")
        certainty = str(llm_answer.get("certainty") or "low")
        cited_sources = cited_source_lines(llm_answer, results)
        if cited_sources and "PDF物理页" not in basis:
            basis = (basis + "；" if basis else "") + "；".join(cited_sources[:2])
        return conclusion, basis, certainty

    if llm_answer and not llm_can_answer and llm_answer.get("note") and not evidence_can_answer:
        basis = f"{confidence['reason']}；模型提示：{llm_answer.get('note')}"
        return "未召回到足够依据，无法可靠回答。", basis, "low"

    if llm_error:
        basis = f"大语言模型未返回可靠结果：{llm_error}"
    else:
        basis = confidence["reason"]

    if results and confidence.get("can_answer"):
        top = results[0]
        sentence = evidence_sentence(question, top.get("text", ""))
        conclusion = sentence or "已召回到相关规范原文，但未能稳定抽取直接结论，请查看引用来源。"
    else:
        conclusion = "未召回到足够依据，无法可靠回答。"
    return conclusion, basis, confidence.get("level", "low")


def understanding_summary(turn: dict) -> str:
    question = str(turn.get("question") or "")
    rewritten = str(turn.get("rewritten") or question)
    if rewritten == question:
        return ""

    context_decision = turn.get("context_decision") or {}
    if context_decision.get("source") == "llm" and context_decision.get("reason"):
        return "上下文判断：" + str(context_decision["reason"])

    reasons = turn.get("rewrite_reasons") or []
    if is_meta_reference_question(question):
        return "已使用上一轮证据辅助回答本轮的条文、页码或复核追问。"
    if turn.get("contextual_followup"):
        if any("最近两轮" in str(reason) for reason in reasons):
            return "已使用最近两轮上下文辅助处理本轮对比追问。"
        if any("上一轮" in str(reason) for reason in reasons):
            return "已使用上一轮上下文辅助回答本轮追问。"
        return "已使用对话上下文辅助检索。"
    return ""


def route_context_decision(question: str, state: ConversationState) -> ContextDecision:
    if not getattr(state, "context_questions", None) and not getattr(state, "rewritten_questions", None):
        return ContextDecision(
            mode="new_topic",
            use_previous_context=False,
            resolved_question=question,
            reason="没有历史上下文，本轮按独立问题处理。",
            source="fallback",
        )
    try:
        decision = route_context_with_llm(question, state, api_key=os.getenv("DEEPSEEK_API_KEY"))
        return normalize_reference_context(question, decision, state)
    except Exception as exc:
        fallback_followup = has_followup_signal(question, state)
        return normalize_reference_context(
            question,
            ContextDecision(
                mode="follow_up" if fallback_followup else "new_topic",
                use_previous_context=fallback_followup,
                resolved_question=question,
                reason=f"上下文路由模型不可用，已使用保守规则兜底：{str(exc)[:120]}",
                source="fallback",
            ),
            state,
        )


def cited_source_lines(llm_answer: dict | None, results: list[dict]) -> list[str]:
    if not llm_answer:
        return []
    lines: list[str] = []
    for citation in llm_answer.get("citations") or []:
        if not isinstance(citation, int):
            continue
        idx = citation - 1
        if 0 <= idx < len(results):
            lines.append(source_brief_display(results[idx]))
    return lines


def render_source_list(results: list[dict], abolished: dict, *, expert_mode: bool) -> None:
    shown = results[:5]
    with st.expander(f"查看引用来源（{len(shown)} 条）", expanded=False):
        for idx, item in enumerate(shown, start=1):
            abolished_info = is_abolished(item, abolished)
            st.markdown(
                f"""
<div class="source-mini">
  <strong>{idx}. {escape(STANDARD_NAMES.get(item.get('standard_id'), item.get('standard_id') or '未知标准'))}</strong>
  <div class="source-meta">{highlight_answer_html(source_brief_display(item))}</div>
  <div class="source-text">{highlight_answer_html(evidence_sentence("", item.get("text", "")) or clean_text(item.get("text", ""), 220))}</div>
</div>
""",
                unsafe_allow_html=True,
            )
            if abolished_info:
                st.warning(
                    f"废止/替代提醒：{abolished_info['source_standard_id']} {abolished_info['source_article']} "
                    f"需核对 {abolished_info['replaced_by_standard_id']}。"
                )
            st.markdown(
                f"""
<div class="source-text">
  <strong>原文片段：</strong>{highlight_answer_html(clean_text(item.get("text", ""), 1800))}
</div>
""",
                unsafe_allow_html=True,
            )
            if expert_mode:
                st.caption("检索细节")
                st.json(
                    {
                        "hybrid_score": item.get("hybrid_score"),
                        "bm25_score": item.get("bm25_score"),
                        "vector_score": item.get("vector_score"),
                        "rrf_score": item.get("fused_score"),
                        "api_rerank_score": item.get("api_rerank_score"),
                        "table_no": item.get("table_no"),
                        "table_score": item.get("table_score"),
                        "chunk_id": item.get("chunk_id"),
                        "row_id": item.get("row_id"),
                        "parent_row_id": item.get("parent_row_id"),
                        "parent_article_no": item.get("parent_article_no"),
                        "inherited_article_no": item.get("inherited_article_no"),
                        "retrieval_query": item.get("retrieval_query"),
                        "policy_notes": item.get("policy_notes"),
                    }
                )


def retrieval_diagnostics(turn: dict, abolished: dict) -> list[str]:
    results = turn.get("results") or []
    confidence = turn.get("confidence") or {}
    query = turn.get("rewritten") or turn.get("question") or ""
    messages: list[str] = []

    if confidence.get("reason"):
        messages.append(str(confidence["reason"]))
    if not results:
        messages.append("没有召回到候选原文，通常需要扩大知识库范围，或把建筑类型、部位、条件写得更具体。")
        return messages

    top = results[0]
    missing_fields = [name for name, ok, _ in field_recall_status(top) if not ok]
    if missing_fields:
        messages.append("首条候选缺少：" + "、".join(missing_fields) + "。字段不完整时，结论需要结合原文复核。")

    top_score = float(top.get("hybrid_score") or 0)
    second_score = float(results[1].get("hybrid_score") or 0) if len(results) > 1 else 0
    if len(results) > 1 and top_score - second_score < 3:
        messages.append("首条候选与后续候选分差较小，存在相邻条文竞争，建议展开引用来源核对适用条款。")

    standard_counts = Counter(item.get("standard_id") or "未知标准" for item in results[:5])
    if len(standard_counts) >= 3:
        messages.append("Top 候选分布在多份规范中，问题可能涉及跨规范场景，建议先限定建筑类型或选定某一份文件。")

    if any(is_abolished(item, abolished) for item in results[:3]):
        messages.append("前列候选包含旧规范废止或替代提醒，最终结论应优先核对现行通用规范。")

    if re.search(r"多少|几|不应大于|不应小于|最大|最小|净宽|面积|距离|高度|耐火", query):
        has_number = any(
            re.search(r"\d+(?:\.\d+)?\s*(?:m|h|㎡|m2|人|层|%)", item.get("text", ""), re.IGNORECASE)
            for item in results[:3]
        )
        if not has_number:
            messages.append("问题涉及数值判断，但前三条候选未直接召回稳定数值字段，当前结论仅可作为检索线索。")

    route = turn.get("route") or {}
    if route.get("reasons"):
        messages.append("范围识别依据：" + "；".join([str(item) for item in route.get("reasons", [])[:2]]))

    return list(dict.fromkeys(messages))


def render_retrieval_diagnostics(turn: dict, abolished: dict) -> None:
    confidence = turn.get("confidence") or {}
    level = confidence.get("level", "low")
    if level == "high":
        return
    diagnostics = retrieval_diagnostics(turn, abolished)
    if not diagnostics:
        return
    with st.expander("检索依据风险与复核提示", expanded=False):
        for item in diagnostics[:6]:
            st.caption(f"- {item}")


LEVEL_RANK = {"low": 0, "medium": 1, "high": 2}
LEVEL_LABEL = {"high": "高", "medium": "中", "low": "低"}


def conservative_model_level(model_level: str, evidence_level: str) -> tuple[str, str | None]:
    model = model_level if model_level in LEVEL_RANK else "low"
    evidence = evidence_level if evidence_level in LEVEL_RANK else "low"
    if LEVEL_RANK[model] <= LEVEL_RANK[evidence]:
        return model, None
    return evidence, f"DeepSeek 原始判断为{LEVEL_LABEL[model]}，但证据置信度为{LEVEL_LABEL[evidence]}，页面按证据等级保守显示。"


def confidence_field_summary(result: dict | None) -> str:
    fields = field_recall_status(result)
    ok_items = [name for name, ok, _ in fields if ok]
    missing = [name for name, ok, _ in fields if not ok]
    parts = []
    if ok_items:
        parts.append("已具备：" + "、".join(ok_items))
    if missing:
        parts.append("缺少：" + "、".join(missing))
    return "；".join(parts) or "无字段信息"


def render_confidence_explanation(turn: dict, abolished: dict, *, model_level_raw: str, model_level_display: str, model_cap_reason: str | None) -> None:
    results = turn.get("results") or []
    confidence = turn.get("confidence") or {}
    llm_answer = turn.get("llm_answer") or {}
    evidence_level = confidence.get("level", "low")

    with st.expander("置信度判定依据", expanded=False):
        st.caption("证据置信度")
        st.caption(f"- 判定：{LEVEL_LABEL.get(evidence_level, evidence_level)}")
        if confidence.get("reason"):
            st.caption(f"- 原因：{confidence['reason']}")
        if results:
            top = results[0]
            st.caption(f"- 首条证据字段：{confidence_field_summary(top)}")
            top_score = float(top.get("hybrid_score") or 0)
            if len(results) > 1:
                second_score = float(results[1].get("hybrid_score") or 0)
                st.caption(f"- Top1/Top2 分差：{top_score - second_score:.2f}")
            abolished_info = is_abolished(top, abolished)
            if abolished_info:
                st.caption(
                    f"- 旧规范提醒：{abolished_info['source_standard_id']} {abolished_info['source_article']} "
                    f"需核对 {abolished_info['replaced_by_standard_id']}。"
                )
        else:
            st.caption("- 未召回候选证据。")

        if llm_answer:
            st.caption("DeepSeek 模型判断置信度")
            st.caption(f"- 原始等级：{LEVEL_LABEL.get(model_level_raw, model_level_raw)}")
            st.caption(f"- 页面显示：{LEVEL_LABEL.get(model_level_display, model_level_display)}")
            if llm_answer.get("confidence_reason"):
                st.caption(f"- 模型给出的原因：{llm_answer.get('confidence_reason')}")
            if llm_answer.get("missing_fields"):
                st.caption("- 模型认为缺失字段：" + "、".join([str(item) for item in llm_answer.get("missing_fields", [])]))
            if llm_answer.get("citations"):
                st.caption("- 模型引用证据编号：" + "、".join([str(item) for item in llm_answer.get("citations", [])]))
            if model_cap_reason:
                st.caption(f"- 保守显示规则：{model_cap_reason}")
        else:
            st.caption("DeepSeek 模型判断置信度")
            st.caption("- 未启用或未返回模型判断。")


def render_chat_turn(turn: dict, abolished: dict, *, expert_mode: bool) -> None:
    question = turn["question"]
    rewritten = turn.get("rewritten") or question
    results = turn.get("results") or []
    confidence = turn.get("confidence") or {"level": "low", "label": "低", "reason": "无置信度信息"}
    llm_answer = turn.get("llm_answer")
    llm_error = turn.get("llm_error")
    conclusion, basis, certainty = final_answer_text(question, results, confidence, llm_answer, llm_error)
    evidence_level = confidence.get("level", "low")
    evidence_label = confidence.get("label", evidence_level)
    badge_class = {
        "high": "answer-badge",
        "medium": "answer-badge mid",
        "low": "answer-badge warn",
    }.get(evidence_level, "answer-badge warn")
    badge_text = f"证据置信度：{evidence_label}"
    model_badge_html = ""
    model_level_raw = certainty if certainty in LEVEL_RANK else "low"
    model_level_display = model_level_raw
    model_cap_reason = None
    if llm_answer:
        model_level_display, model_cap_reason = conservative_model_level(model_level_raw, evidence_level)
        model_badge_class = "answer-badge model" if model_level_display in {"high", "medium"} else "answer-badge model warn"
        model_label = LEVEL_LABEL.get(model_level_display, "低")
        raw_suffix = ""
        if model_cap_reason:
            raw_suffix = f"（原始：{LEVEL_LABEL.get(model_level_raw, model_level_raw)}）"
        model_badge_html = f'<span class="{model_badge_class}">模型判断置信度：{escape(model_label + raw_suffix)}</span>'

    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
    st.markdown('<div class="user-turn-label">提问</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="user-bubble">{escape(question)}</div>', unsafe_allow_html=True)
    summary = understanding_summary(turn)
    if summary:
        st.markdown(f'<div class="understood">{escape(summary)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="user-turn-label">规范顾问</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            f"""
<div class="answer-main">
  <div class="answer-title">{highlight_answer_html(conclusion)}</div>
  <div class="answer-basis">依据：{highlight_answer_html(basis or confidence.get("reason", ""))}</div>
  <div class="confidence-row">
    <span class="{badge_class}">{escape(badge_text)}</span>
    {model_badge_html}
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        if llm_answer and llm_answer.get("missing_fields"):
            st.caption("缺失字段：" + "、".join([str(item) for item in llm_answer.get("missing_fields", [])]))
        render_confidence_explanation(
            turn,
            abolished,
            model_level_raw=model_level_raw,
            model_level_display=model_level_display,
            model_cap_reason=model_cap_reason,
        )
        if expert_mode and turn.get("contextual_followup"):
            with st.expander("上下文判断依据", expanded=False):
                decision = turn.get("context_decision") or {}
                st.caption("这不是问题到答案的映射，只是决定本轮是否借用上一轮检索证据。")
                st.caption(f"路由来源：{decision.get('source', 'unknown')}；模式：{decision.get('mode', 'unknown')}")
                st.caption("触发原因：" + ("；".join([str(item) for item in turn.get("rewrite_reasons", [])]) or "命中通用追问信号"))
                st.caption(f"继承证据数量：{turn.get('context_evidence_count', 0)}")
                if rewritten != question:
                    st.caption(f"检索补全文：{rewritten}")
        render_retrieval_diagnostics(turn, abolished)
        if results:
            render_source_list(results, abolished, expert_mode=expert_mode)
    st.markdown("</div>", unsafe_allow_html=True)


def append_query_turn(
    query: str,
    *,
    scope: dict,
    top_k: int,
    expert_mode: bool,
    use_api_rerank: bool,
    registry: dict,
    abolished: dict,
    vector_config: dict,
) -> None:
    if "conversation_state" not in st.session_state:
        st.session_state.conversation_state = ConversationState()
    if "chat_turns" not in st.session_state:
        st.session_state.chat_turns = []

    context_decision = route_context_decision(query, st.session_state.conversation_state)
    contextual_followup = bool(context_decision.use_previous_context)
    rewritten = context_decision.resolved_question if contextual_followup and context_decision.resolved_question else query
    rewrite_reasons = [context_decision.reason or ("上下文路由判定为追问" if contextual_followup else "上下文路由判定为新问题")]
    normalized_ifc_query = normalize_ifc_review_query(rewritten)
    if normalized_ifc_query != rewritten:
        rewritten = normalized_ifc_query
        rewrite_reasons.append("IFC 审查问题已归一化：保留模型事实字段，移除族名、实例编号和尺寸字符串噪声。")
    remember_question(st.session_state.conversation_state, query, rewritten)
    routed_query = rewritten
    forced_standard = scope.get("standard_id") or ""
    if forced_standard:
        routed_query = f"{forced_standard} {rewritten}"

    route = infer_route(routed_query, registry)
    results = hybrid_search(
        routed_query,
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        use_api_rerank=use_api_rerank,
        rerank_top_n=top_k,
        final_top_k=top_k,
    )
    context_results = []
    if contextual_followup:
        context_sets = 2 if is_meta_reference_question(query) else 1
        context_results = recent_evidence(st.session_state.conversation_state, sets=context_sets, max_items=top_k)
        context_results = rank_context_evidence(query, context_results)
        if context_results:
            if prefers_context_evidence(query):
                results = merge_evidence_results(context_results, results, max_items=max(top_k, 10))[:top_k]
            else:
                results = merge_evidence_results(results, context_results, max_items=max(top_k, 10))[:top_k]
            rewrite_reasons = list(rewrite_reasons) + ["沿用上一轮可审查证据"]
    update_state_from_results(st.session_state.conversation_state, results)
    confidence = confidence_from_results(results, route.primary + route.secondary, abolished, routed_query)

    llm_answer = None
    llm_error = None
    if os.getenv("DEEPSEEK_API_KEY"):
        try:
            llm_answer = generate_answer(
                query,
                results[:5],
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                retrieval_question=rewritten,
                is_followup=contextual_followup,
                is_reference_question=is_meta_reference_question(query) or prefers_context_evidence(query),
            )
        except (LLMAnswerError, ValueError, KeyError, IndexError) as exc:
            llm_error = str(exc)[:500]

    st.session_state.chat_turns.append(
        {
            "question": query,
            "rewritten": rewritten,
            "rewrite_reasons": rewrite_reasons,
            "scope": scope.get("label", "自动识别"),
            "scope_standard_id": forced_standard,
            "route": {"primary": route.primary, "secondary": route.secondary, "reasons": route.reasons},
            "confidence": confidence,
            "results": results,
            "contextual_followup": contextual_followup,
            "context_decision": {
                "mode": context_decision.mode,
                "use_previous_context": context_decision.use_previous_context,
                "resolved_question": context_decision.resolved_question,
                "reason": context_decision.reason,
                "context_used": context_decision.context_used,
                "source": context_decision.source,
            },
            "context_evidence_count": len(context_results),
            "llm_answer": llm_answer,
            "llm_error": llm_error,
            "vector_model": vector_config.get("embedding_model"),
            "expert_mode": expert_mode,
        }
    )
    history = st.session_state.setdefault("conversation_history", [])
    history.append(
        {
            "question": query,
            "rewritten": rewritten,
            "confidence": confidence["level"],
            "top_standard": results[0].get("standard_id") if results else None,
            "top_article": results[0].get("article_no") if results else None,
        }
    )
    if not is_meta_reference_question(query):
        remember_evidence(st.session_state.conversation_state, results)


def render_library_page(registry: dict, vector_config: dict, vector_metas: list[dict]) -> None:
    standards = registry.get("standards", {})
    st.markdown('<div class="module-page">', unsafe_allow_html=True)
    st.markdown(
        """
<div class="module-card">
  <strong>规范库</strong>
  <div class="starter-text">管理当前可检索的规范文档、标准号、文件名和知识片段状态。这里是数据资产视图，不展示研发测试指标。</div>
</div>
""",
        unsafe_allow_html=True,
    )
    rows = []
    source_by_standard = {}
    for meta in vector_metas:
        if meta.get("standard_id") and meta.get("source_file"):
            source_by_standard.setdefault(meta.get("standard_id"), meta.get("source_file"))
    for standard_id, item in sorted(standards.items(), key=lambda pair: (-int(pair[1].get("priority", 0)), pair[0])):
        rows.append(
            {
                "规范名称": item.get("title", STANDARD_NAMES.get(standard_id, standard_id)),
                "标准号": standard_id,
                "文件名": source_by_standard.get(standard_id) or item.get("file") or f"{standard_id}.pdf",
                "优先级": item.get("priority", ""),
            }
        )
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    st.markdown(
        f"""
<div class="module-card">
  <strong>知识库摘要</strong>
  <div class="starter-text">知识片段：{escape(str(vector_config.get("chunks", "未知")))}；向量模型：{escape(str(vector_config.get("embedding_model", "未知")))}。</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_history_page() -> None:
    history = st.session_state.get("conversation_history") or []
    st.markdown('<div class="module-page">', unsafe_allow_html=True)
    st.markdown(
        """
<div class="module-card">
  <strong>查询历史</strong>
  <div class="starter-text">用于复盘用户问过什么、系统命中了哪个标准和条文。当前先保留本次会话历史，不做账号级持久化。</div>
</div>
""",
        unsafe_allow_html=True,
    )
    if history:
        st.dataframe(
            [
                {
                    "问题": item.get("question"),
                    "改写后问题": item.get("rewritten"),
                    "证据置信度": item.get("confidence"),
                    "首条标准": item.get("top_standard"),
                    "首条条文": item.get("top_article"),
                }
                for item in history
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("当前还没有查询历史。")
    st.markdown("</div>", unsafe_allow_html=True)


def render_status_page(registry: dict, vector_config: dict) -> None:
    st.markdown('<div class="module-page">', unsafe_allow_html=True)
    st.markdown(
        """
<div class="module-card">
  <strong>系统状态</strong>
  <div class="starter-text">展示面向用户的运行状态和能力边界，不展示内部迭代记录或测试集指标。</div>
</div>
""",
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    cols[0].metric("规范数量", len(registry.get("standards", {})))
    cols[1].metric("知识片段", vector_config.get("chunks", "未知"))
    cols[2].metric("DeepSeek", "已配置" if os.getenv("DEEPSEEK_API_KEY") else "未配置")
    st.markdown(
        """
<div class="module-card">
  <strong>当前能力边界</strong>
  <div class="starter-text">规范问答可以独立使用；BIM 模型审查需要先解析 IFC，再补充建筑用途、耐火等级、自动灭火系统等模型无法稳定识别的项目条件。</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="建筑消防规范智能问答", layout="wide")
    inject_css()
    ensure_env()

    registry, abolished = get_policy()
    try:
        _, vector_metas, vector_config = load_vectorstore()
    except Exception as exc:
        st.error(f"规范知识库不可用：{exc}")
        st.stop()

    expert_mode, use_api_rerank, top_k = render_sidebar(registry, vector_config)
    if "chat_turns" not in st.session_state:
        st.session_state.chat_turns = []

    active_module = st.session_state.get("active_module", "规范问答")
    render_mode_nav(active_module)

    if active_module == "规范库":
        render_library_page(registry, vector_config, vector_metas)
        return
    if active_module == "查询历史":
        render_history_page()
        return
    if active_module == "系统状态":
        render_status_page(registry, vector_config)
        return
    if active_module == "BIM 模型审查":
        ifc_scope = current_scope_option(registry, vector_metas, "ifc_scope_label")
        for block_id in sorted(["ifc_scope", "ifc_panel"], key=ui_order):
            if not ui_visible(block_id):
                continue
            if block_id == "ifc_scope":
                st.markdown(
                    f'<div class="module-subtitle" style="{ui_width_style(block_id)} {ui_text_style(block_id, "body")}">上传 IFC 后先看模型事实，再补充项目条件，最后进入消防合规审查。这里不和独立规范问答混在一起。</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f'<div style="{ui_width_style(block_id)}">', unsafe_allow_html=True)
                ifc_scope = render_scope_selector(registry, vector_metas, key="ifc_scope_label", block_id=block_id)
                st.markdown("</div>", unsafe_allow_html=True)
            elif block_id == "ifc_panel":
                ui_block_label(block_id)
                st.markdown(f'<div style="{ui_width_style(block_id)}">', unsafe_allow_html=True)
                render_ifc_panel(
                    scope=ifc_scope,
                    top_k=top_k,
                    use_api_rerank=use_api_rerank,
                    registry=registry,
                    abolished=abolished,
                    vector_config=vector_config,
                    expert_mode=expert_mode,
                )
                st.markdown("</div>", unsafe_allow_html=True)
        return

    scope = current_scope_option(registry, vector_metas, "qa_scope_label")

    if st.session_state.get("pending_prompt"):
        prompt = st.session_state.pop("pending_prompt")
        if vector_config.get("embedding_provider") == "siliconflow" and not os.getenv("SILICONFLOW_API_KEY"):
            st.error("缺少 SILICONFLOW_API_KEY，无法使用当前知识库。")
            st.stop()
        with st.spinner("正在检索规范依据并生成回答..."):
            append_query_turn(
                prompt,
                scope=scope,
                top_k=top_k,
                expert_mode=expert_mode,
                use_api_rerank=use_api_rerank,
                registry=registry,
                abolished=abolished,
                vector_config=vector_config,
            )
        st.rerun()

    submitted = False
    prompt = ""
    for block_id in sorted(["qa_scope", "qa_intro", "qa_templates", "qa_answers", "qa_input"], key=ui_order):
        if not ui_visible(block_id):
            continue
        if block_id == "qa_scope":
            st.markdown(
                f'<div class="module-subtitle" style="{ui_width_style(block_id)} {ui_text_style(block_id, "body")}">独立规范问答不需要上传模型。系统只负责检索规范、生成可核查回答，并把引用来源折叠展示。</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f'<div style="{ui_width_style(block_id)}">', unsafe_allow_html=True)
            scope = render_scope_selector(registry, vector_metas, key="qa_scope_label", block_id=block_id)
            st.markdown("</div>", unsafe_allow_html=True)
        elif block_id == "qa_intro" and not st.session_state.chat_turns:
            ui_block_label(block_id)
            st.markdown(
                f"""
<div class="starter-card" style="{ui_width_style(block_id)}">
  <div class="starter-title" style="{ui_text_style(block_id, "title")}">选择一种问题类型，或直接输入。</div>
  <div class="starter-text" style="{ui_text_style(block_id, "body")}">问题里尽量写清建筑场景、部位、条件和需要判断的数值；系统会把召回依据、原文片段和 PDF 物理页折叠展示。</div>
</div>
""",
                unsafe_allow_html=True,
            )
        elif block_id == "qa_templates" and not st.session_state.chat_turns:
            ui_block_label(block_id)
            st.markdown(f'<div style="{ui_width_style(block_id)}">', unsafe_allow_html=True)
            cols = st.columns(2)
            for idx, template in enumerate(QUESTION_TEMPLATES):
                col = cols[idx % 2]
                with col:
                    st.markdown(
                        f"""
<div class="starter-card">
  <div class="starter-title" style="{ui_text_style(block_id, "title")}">{escape(template["title"])}</div>
  <div class="starter-text" style="{ui_text_style(block_id, "body")}">{escape(template["hint"])}</div>
</div>
""",
                        unsafe_allow_html=True,
                    )
                    if st.button("使用这个问题模板", key=f"template_{idx}", use_container_width=True):
                        example = template["query"]
                        st.session_state.pending_prompt = example
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        elif block_id == "qa_answers":
            ui_block_label(block_id)
            st.markdown(f'<div style="{ui_width_style(block_id)}">', unsafe_allow_html=True)
            for turn in st.session_state.chat_turns:
                render_chat_turn(turn, abolished, expert_mode=expert_mode)
            st.markdown("</div>", unsafe_allow_html=True)
        elif block_id == "qa_input":
            ui_block_label(block_id)
            st.markdown(f'<div class="query-input-panel" style="{ui_width_style(block_id)}">', unsafe_allow_html=True)
            with st.form("qa_query_form", clear_on_submit=True):
                input_cols = st.columns([8, 1.35])
                with input_cols[0]:
                    prompt = st.text_input(
                        "输入你的建筑规范问题",
                        placeholder="输入规范查询问题，例如：商业综合体中庭四周的防火卷帘设置有什么具体要求？",
                        label_visibility="collapsed",
                    )
                with input_cols[1]:
                    submitted = st.form_submit_button("发送", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    if submitted and clean_text(prompt):
        if vector_config.get("embedding_provider") == "siliconflow" and not os.getenv("SILICONFLOW_API_KEY"):
            st.error("缺少 SILICONFLOW_API_KEY，无法使用当前知识库。")
            st.stop()
        with st.spinner("正在检索规范依据并生成回答..."):
            append_query_turn(
                prompt,
                scope=scope,
                top_k=top_k,
                expert_mode=expert_mode,
                use_api_rerank=use_api_rerank,
                registry=registry,
                abolished=abolished,
                vector_config=vector_config,
            )
        st.rerun()


if __name__ == "__main__":
    main()
