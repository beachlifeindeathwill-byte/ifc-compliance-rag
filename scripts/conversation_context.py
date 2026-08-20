from __future__ import annotations

import re
from dataclasses import dataclass, field

from context_router import ContextDecision


BUILDING_TERMS = [
    "高层医疗建筑",
    "高层公共建筑",
    "普通高层公共建筑",
    "单多层建筑",
    "单、多层建筑",
    "高层民用建筑",
    "地下汽车库",
    "半地下汽车库",
    "汽车库",
    "医疗建筑",
    "普通多层住宅",
    "多层住宅",
    "高层住宅",
    "住宅建筑",
    "住宅",
    "公共建筑",
    "民用建筑",
    "地下或半地下建筑",
    "地下建筑",
]

TOPIC_TERMS = [
    "疏散楼梯净宽",
    "疏散楼梯的最小净宽",
    "疏散楼梯宽度",
    "疏散走道净宽",
    "疏散门净宽",
    "疏散出口门净宽",
    "防火分区最大允许建筑面积",
    "防火分区面积",
    "防火分区",
    "自动灭火系统",
    "自动扶梯",
    "防火墙",
    "疏散楼梯数量",
    "规范依据",
]

CORRECTION_TERMS = ["不对", "不是", "我问的是", "更正", "改成"]
REFERENCE_TERMS = ["那", "这个", "这种情况", "这种情况下", "刚才", "前面", "上一个", "第一个问题", "其"]
REFERENCE_TERMS.extend(["这两个", "两个数值", "上述", "对应", "分别"])
CONDITIONAL_TERMS = ["如果", "是否", "会不会", "能不能", "可不可以", "可以吗", "也是", "也要", "同样", "放宽", "这里", "的话"]

META_REFERENCE_PATTERNS = [
    r"(?:依据|条文依据|出处|来源|规范依据|标准号|条文号|表号|页码|PDF).{0,12}(?:是什么|是哪|是啥|来自|哪一条|哪条|哪个|列出来|多少)",
    r"(?:来自|出自).{0,10}(?:哪一条|哪条|哪个|什么|哪里|何处)",
    r"哪一条|哪条原文|哪一款|哪条",
    r"(?:这个|上述|刚才|上一轮|前面|第一个问题).{0,24}(?:结论|依据|条文|要求|计算|数值|结果|说法)",
    r"(?:该要求|该结论|该数值|该条文|该计算结果|该结果|该说法)",
    r"为什么.{0,16}(?:数值|要求|规定|标准|条文|结果|两个|两者).{0,8}(?:不同|不一样|差异|区别)",
    r"为什么.{0,12}(?:不一样|有差异|有区别)",
    r"这两个|两个问题|两个数值|分别给我|分别列出|分别说明",
    r"人工复核|复核的地方|需要复核",
    r"回到|返回|说回|接着刚才|继续刚才|第一个问题",
]

BACK_REFERENCE_PATTERNS = [
    r"哪一条|哪条原文|哪一款|哪条",
    r"(?:来自|出自).{0,10}(?:哪|什么|哪里|何处)",
    r"(?:依据|条文依据|出处|来源|规范依据|标准号|条文号|表号|页码|PDF).{0,12}(?:是什么|是哪|是啥|列出来|给出|多少)",
    r"(?:对应|列出|写明|提供).{0,8}(?:标准|条文|表号|页码|依据|出处)",
]

COMPARISON_PATTERNS = [
    r"为什么.{0,16}(?:数值|要求|规定|标准|条文|结果|两个|两者).{0,8}(?:不同|不一样|差异|区别)",
    r"为什么.{0,12}(?:不一样|有差异|有区别)",
    r"这两个|两个问题|两个数值|各自",
    r"(?:和|与).{0,12}(?:相比|区别|差异)",
]

RETURN_PATTERNS = [
    r"回到|返回|说回|接着刚才|继续刚才|回到刚才|第一个问题",
]

CONTEXT_ANCHOR_PATTERNS = [
    r"这个|这些",
    r"该(?:要求|结论|数值|条文|计算结果|结果|说法)",
    r"上述|刚才|上一轮|前面|前面提到的|第一个问题",
    r"这两个|两个问题|两个数值",
    r"回到|返回|说回|接着刚才|继续刚才|回到刚才",
]

TARGET_TERMS = [
    "净宽",
    "宽度",
    "面积",
    "距离",
    "耐火极限",
    "数量",
    "个数",
    "要求",
    "条件",
    "设置",
    "划分",
    "计算",
    "防火性能",
    "联动",
]


@dataclass
class ConversationState:
    building_terms: list[str] = field(default_factory=list)
    topic_terms: list[str] = field(default_factory=list)
    standard_ids: list[str] = field(default_factory=list)
    article_nos: list[str] = field(default_factory=list)
    context_questions: list[str] = field(default_factory=list)
    rewritten_questions: list[str] = field(default_factory=list)
    evidence_sets: list[list[dict]] = field(default_factory=list)


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def extract_terms(text: str, candidates: list[str]) -> list[str]:
    compact = re.sub(r"\s+", "", text or "")
    hits = []
    for term in candidates:
        if re.sub(r"\s+", "", term) in compact:
            hits.append(term)
    return _dedupe(sorted(hits, key=len, reverse=True))


def infer_topic_terms(question: str) -> list[str]:
    hits = extract_terms(question, TOPIC_TERMS)
    compact = re.sub(r"\s+", "", question or "")
    if "疏散楼梯" in compact and "净宽" in compact:
        hits.append("疏散楼梯净宽")
    if "疏散走道" in compact and "净宽" in compact:
        hits.append("疏散走道净宽")
    if "疏散门" in compact and "净宽" in compact:
        hits.append("疏散门净宽")
    if "防火分区" in compact and ("面积" in compact or "划分" in compact):
        hits.append("防火分区面积")
    if "依据" in compact or "哪一条" in compact:
        hits.append("规范依据")
    return _dedupe(hits)


def has_reference(question: str) -> bool:
    return any(term in question for term in REFERENCE_TERMS)


def has_correction(question: str) -> bool:
    return any(term in question for term in CORRECTION_TERMS)


def is_self_contained_question(question: str) -> bool:
    compact = re.sub(r"\s+", "", question or "")
    has_scope = bool(extract_terms(question, BUILDING_TERMS)) or bool(re.search(r"GB\s*\d{5}(?:-\d{4})?", question or "", re.I))
    has_target = bool(infer_topic_terms(question)) or any(term in compact for term in TARGET_TERMS)
    has_design_value = bool(re.search(r"\d+(?:\.\d+)?\s*(?:m|米|h|㎡|m2|m²|个|人|辆|%)", question or "", re.I))
    generic_building_scope = "建筑" in compact and has_target
    return (has_scope and has_target) or (has_target and has_design_value) or generic_building_scope


def has_explicit_scope_and_target(question: str) -> bool:
    """Return whether the current turn independently names both scope and review target."""
    compact = re.sub(r"\s+", "", question or "")
    has_scope = bool(extract_terms(question, BUILDING_TERMS)) or bool(
        re.search(r"GB\s*\d{5}(?:-\d{4})?", question or "", re.I)
    )
    has_target = bool(infer_topic_terms(question)) or any(term in compact for term in TARGET_TERMS)
    return has_scope and has_target


def ensure_state_fields(state: ConversationState) -> None:
    for name in ["building_terms", "topic_terms", "standard_ids", "article_nos", "context_questions", "rewritten_questions", "evidence_sets"]:
        if not hasattr(state, name):
            setattr(state, name, [])


def is_meta_reference_question(question: str) -> bool:
    compact = re.sub(r"\s+", "", question or "")
    return any(re.search(pattern, compact) for pattern in META_REFERENCE_PATTERNS)


def asks_back_reference(question: str) -> bool:
    compact = re.sub(r"\s+", "", question or "")
    return any(re.search(pattern, compact) for pattern in BACK_REFERENCE_PATTERNS)


def is_comparison_question(question: str) -> bool:
    compact = re.sub(r"\s+", "", question or "")
    return any(re.search(pattern, compact) for pattern in COMPARISON_PATTERNS)


def has_return_signal(question: str) -> bool:
    compact = re.sub(r"\s+", "", question or "")
    return any(re.search(pattern, compact) for pattern in RETURN_PATTERNS)


def has_context_anchor(question: str) -> bool:
    compact = re.sub(r"\s+", "", question or "")
    return any(re.search(pattern, compact) for pattern in CONTEXT_ANCHOR_PATTERNS)


def prefers_context_evidence(question: str) -> bool:
    return is_meta_reference_question(question) or asks_back_reference(question) or any(
        term in question for term in ["这个要求", "该要求", "这个结论", "上述要求", "上述结论"]
    )


def _question_signals(question: str) -> tuple[list[str], list[str], list[str], list[str]]:
    compact = re.sub(r"\s+", "", question or "")
    buildings = extract_terms(question, BUILDING_TERMS)
    topics = infer_topic_terms(question)
    numbers = list(dict.fromkeys(re.findall(r"\d+(?:\.\d+)?", compact)))
    standards = list(dict.fromkeys(re.findall(r"GB\s*\d{5}(?:-\d{4})?", question or "", re.I)))
    return buildings, topics, numbers, standards


def rank_history_questions(question: str, state: ConversationState, limit: int = 3) -> list[str]:
    ensure_state_fields(state)
    candidates = list(state.context_questions or state.rewritten_questions)
    if not candidates:
        return []
    buildings, topics, numbers, standards = _question_signals(question)
    if not buildings and not topics and not numbers and not standards:
        return candidates[-limit:][::-1]

    scored: list[tuple[int, int, str]] = []
    for index, candidate in enumerate(candidates):
        compact = re.sub(r"\s+", "", candidate)
        score = 0
        for term in buildings:
            if term in compact:
                score += 4
        for term in topics:
            if term in compact:
                score += 3
        for number in numbers:
            if number in compact:
                score += 2
        for standard in standards:
            if standard in candidate:
                score += 2
        scored.append((score, index, candidate))
    scored.sort(key=lambda item: (-item[0], -item[1]))
    return [item[2] for item in scored[:limit] if item[0] > 0] or candidates[-1:]


def build_reference_retrieval_question(
    question: str,
    state: ConversationState,
) -> tuple[str, list[str], list[str]]:
    ensure_state_fields(state)
    history = list(state.context_questions or state.rewritten_questions)
    if not history:
        return question, ["没有可追溯的历史问题"], []

    if "第一个问题" in question:
        referenced = [history[0]]
        reasons = ["补入第一个问题作为长距离回溯上下文"]
    elif is_comparison_question(question):
        referenced = history[-2:]
        reasons = ["补入最近两轮问题作为对比上下文"]
    else:
        ranked = rank_history_questions(question, state, limit=2)
        referenced = ranked or [history[-1]]
        reasons = ["按本轮明确的建筑对象、数值或检查项匹配历史问题"] if ranked else ["补入最近一轮问题作为证据定位上下文"]

    clean_refs = _dedupe(referenced)
    rewritten = "；".join(clean_refs)
    return rewritten, reasons, clean_refs


def rank_context_evidence(question: str, context_results: list[dict]) -> list[dict]:
    if not context_results:
        return []
    buildings, topics, numbers, standards = _question_signals(question)
    if not buildings and not topics and not numbers and not standards:
        return list(context_results)

    scored: list[tuple[int, int, dict]] = []
    for index, item in enumerate(context_results):
        compact = re.sub(
            r"\s+",
            "",
            " ".join(
                [
                    str(item.get("standard_id") or ""),
                    str(item.get("source_file") or ""),
                    str(item.get("article_no") or ""),
                    str(item.get("text") or ""),
                ]
            ),
        )
        score = 0
        for term in buildings:
            if term in compact:
                score += 4
        for term in topics:
            if term in compact:
                score += 3
        for number in numbers:
            if number in compact:
                score += 2
        for standard in standards:
            if standard in compact:
                score += 2
        scored.append((score, index, item))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored]


def normalize_reference_context(
    question: str,
    decision: ContextDecision | None,
    state: ConversationState | None = None,
) -> ContextDecision:
    ensure_state_fields(state) if state is not None else None
    has_history = bool(state and (state.context_questions or state.rewritten_questions))
    source = decision.source if decision else "guard"
    if not has_history:
        return decision or ContextDecision(
            mode="new_topic",
            use_previous_context=False,
            resolved_question=question,
            reason="没有可追溯的历史上下文。",
            source="guard",
        )

    if is_meta_reference_question(question) and (
        has_context_anchor(question) or has_return_signal(question)
    ):
        if asks_back_reference(question):
            mode = "evidence_request"
            base_reason = "用户追问上一轮结论、条文或数值依据，已按证据追溯处理。"
        elif is_comparison_question(question):
            mode = "comparison"
            base_reason = "用户要求对比历史结论或数值，已按对比上下文处理。"
        else:
            mode = "follow_up"
            base_reason = "用户主动回到之前主题或结论，已按连续上下文处理。"
        resolved, reasons, referenced = build_reference_retrieval_question(question, state)
        return ContextDecision(
            mode=mode,
            use_previous_context=True,
            resolved_question=resolved,
            reason=f"{base_reason}{' '.join(reasons) if reasons else ''}",
            context_used=referenced or ["上一轮问题", "上一轮证据"],
            source=source,
        )

    if has_return_signal(question):
        ranked = rank_history_questions(question, state, limit=1)
        resolved = question
        reasons = []
        if not has_explicit_scope_and_target(question) and ranked:
            resolved = f"{ranked[0]}；{question}"
            reasons.append("补入匹配的历史问题作为场景上下文")
        return ContextDecision(
            mode="follow_up",
            use_previous_context=True,
            resolved_question=resolved,
            reason="用户明确回到之前主题，已保留对话上下文。" + " ".join(reasons),
            context_used=ranked or ["上一轮问题"],
            source=source,
        )

    if (
        decision
        and decision.use_previous_context
        and not is_meta_reference_question(question)
        and has_explicit_scope_and_target(question)
    ):
        return ContextDecision(
            mode="new_topic",
            use_previous_context=False,
            resolved_question=question,
            reason="本轮已明确给出建筑范围和检查对象，按独立检索处理。",
            context_used=[],
            source="guard",
        )
    return decision


def has_followup_signal(question: str, state: ConversationState) -> bool:
    ensure_state_fields(state)
    if not state.context_questions and not state.rewritten_questions:
        return False
    if has_reference(question) or has_correction(question) or asks_back_reference(question):
        return True
    if is_self_contained_question(question):
        return False
    return any(term in question for term in CONDITIONAL_TERMS)


def update_state_from_results(state: ConversationState, results: list[dict]) -> None:
    ensure_state_fields(state)
    for item in results[:3]:
        standard_id = item.get("standard_id")
        article_no = item.get("article_no")
        if standard_id:
            state.standard_ids = _dedupe([standard_id] + state.standard_ids)
        if article_no:
            state.article_nos = _dedupe([article_no] + state.article_nos)


def remember_evidence(state: ConversationState, results: list[dict], *, max_sets: int = 6, max_items: int = 5) -> None:
    ensure_state_fields(state)
    compact_results = []
    for item in results[:max_items]:
        compact_results.append(
            {
                "standard_id": item.get("standard_id"),
                "source_file": item.get("source_file"),
                "article_no": item.get("article_no"),
                "page": item.get("page"),
                "text": item.get("text"),
                "chunk_id": item.get("chunk_id"),
                "row_id": item.get("row_id"),
                "hybrid_score": item.get("hybrid_score"),
                "policy_notes": item.get("policy_notes"),
            }
        )
    if compact_results:
        state.evidence_sets.append(compact_results)
        state.evidence_sets = state.evidence_sets[-max_sets:]


def recent_evidence(state: ConversationState, *, sets: int = 1, max_items: int = 5) -> list[dict]:
    ensure_state_fields(state)
    selected = state.evidence_sets[-sets:] if sets > 0 else []
    merged: list[dict] = []
    seen: set[str] = set()

    def add_items(items: list[dict]) -> None:
        for item in items:
            key = str(item.get("chunk_id") or item.get("row_id") or (item.get("standard_id"), item.get("article_no"), item.get("page")))
            if key in seen:
                continue
            seen.add(key)
            updated = dict(item)
            updated["policy_notes"] = list(updated.get("policy_notes") or []) + ["conversation-evidence: 继承上一轮证据"]
            merged.append(updated)
            if len(merged) >= max_items:
                return

    if len(selected) <= 1:
        add_items(reversed(selected[0]) if selected else [])
        return merged

    # When multiple turns are requested, keep at least one item per turn before
    # filling the quota, so an unrelated recent turn cannot evict the referenced one.
    for evidence_set in selected:
        for item in evidence_set:
            key = str(item.get("chunk_id") or item.get("row_id") or (item.get("standard_id"), item.get("article_no"), item.get("page")))
            if key in seen:
                continue
            seen.add(key)
            updated = dict(item)
            updated["policy_notes"] = list(updated.get("policy_notes") or []) + ["conversation-evidence: 继承上一轮证据"]
            merged.append(updated)
            break
        if len(merged) >= max_items:
            return merged

    for evidence_set in reversed(selected):
        for item in evidence_set:
            key = str(item.get("chunk_id") or item.get("row_id") or (item.get("standard_id"), item.get("article_no"), item.get("page")))
            if key in seen:
                continue
            add_items([item])
            if len(merged) >= max_items:
                return merged
    return merged


def merge_evidence_results(context_results: list[dict], fresh_results: list[dict], *, max_items: int = 10) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    for source in (context_results, fresh_results):
        for item in source:
            key = str(item.get("chunk_id") or item.get("row_id") or (item.get("standard_id"), item.get("article_no"), item.get("page")))
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= max_items:
                return merged
    return merged


def update_state_from_question(state: ConversationState, question: str) -> None:
    ensure_state_fields(state)
    buildings = extract_terms(question, BUILDING_TERMS)
    topics = infer_topic_terms(question)
    if buildings:
        state.building_terms = _dedupe(buildings + state.building_terms)
    if topics:
        state.topic_terms = _dedupe(topics + state.topic_terms)


def remember_question(state: ConversationState, question: str, rewritten: str | None = None) -> None:
    ensure_state_fields(state)
    update_state_from_question(state, question)
    if not is_meta_reference_question(question):
        state.context_questions.append(question)
    state.rewritten_questions.append(rewritten or question)


def rewrite_followup(question: str, state: ConversationState) -> tuple[str, list[str]]:
    ensure_state_fields(state)
    update_state_from_question(state, question)
    reasons: list[str] = []
    additions: list[str] = []

    correction = has_correction(question)
    referenced = has_reference(question)
    comparison_reference = any(term in question for term in ["这两个", "两个问题", "两个数值", "分别"])

    back_reference = asks_back_reference(question)
    if comparison_reference and (state.context_questions or state.rewritten_questions):
        history = state.context_questions[-2:] if state.context_questions else state.rewritten_questions[-2:]
        additions.extend(history)
        reasons.append("补入最近两轮问题作为对比上下文")
    elif "第一个问题" in question and (state.context_questions or state.rewritten_questions):
        history = state.context_questions or state.rewritten_questions
        additions.append(history[0])
        reasons.append("补入第一轮问题作为回溯上下文")
    elif has_followup_signal(question, state) and (state.context_questions or state.rewritten_questions):
        history = state.context_questions or state.rewritten_questions
        additions.append(history[-1])
        reasons.append("补入上一轮问题作为证据定位上下文")

    if back_reference and not comparison_reference:
        additions.extend(state.standard_ids[:1])
        additions.extend(state.article_nos[:2])
        reasons.append("补入上一轮命中的标准和条文号")

    clean_additions = [item for item in _dedupe(additions) if item and item not in question]
    rewritten = question
    if clean_additions:
        rewritten = f"{' '.join(clean_additions)}；{question}"
    remember_question(state, question, rewritten)
    return rewritten, _dedupe(reasons)
