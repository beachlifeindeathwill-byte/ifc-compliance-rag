from __future__ import annotations

import re

from field_extraction import clean_text
from query_intent import detect_query_intent
from retrieval_policy import is_abolished


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


def confidence_from_results(
    results: list[dict],
    route_primary: list[str],
    abolished: dict,
    question: str = "",
) -> dict:
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
            "reason": "已召回到明确旧版或专项规范条文，但该条文存在废止或替代提醒，最终结论需要结合现行通用规范复核。",
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
