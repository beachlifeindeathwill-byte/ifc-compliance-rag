from __future__ import annotations

import json
import re
from typing import Any


UNCERTAINTY_MARKERS = ("可能", "疑似", "暂定", "待确认", "不确定", "未知", "尚未确定", "未确定", "也许", "大概")


def _clean_text(value: Any, limit: int = 500) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _string_list(value: Any, *, limit: int = 12) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = _clean_text(item)
        if text and text not in items:
            items.append(text)
        if len(items) >= limit:
            break
    return items


def classify_user_conditions(project_context: str) -> tuple[list[str], list[str]]:
    confirmed: list[str] = []
    uncertain: list[str] = []
    for clause in re.split(r"[。；;\n]+", _clean_text(project_context, 1600)):
        text = clause.strip(" ，,")
        if not text:
            continue
        target = uncertain if any(marker in text for marker in UNCERTAINTY_MARKERS) else confirmed
        if text not in target:
            target.append(text)
    return confirmed, uncertain


def normalize_project_profile(raw: Any, project_context: str = "") -> dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}
    explicit_confirmed, explicit_uncertain = classify_user_conditions(project_context)
    questions: list[dict[str, Any]] = []
    for item in data.get("review_questions") or []:
        if not isinstance(item, dict):
            continue
        title = _clean_text(item.get("title"), 50)
        question = _clean_text(item.get("question"), 600)
        if not title or not question:
            continue
        questions.append(
            {
                "title": title,
                "question": question,
                "reason": _clean_text(item.get("reason"), 240),
                "facts_used": _string_list(item.get("facts_used"), limit=8),
            }
        )
        if len(questions) >= 6:
            break

    confirmed_conditions = _string_list(data.get("confirmed_conditions"))
    uncertain_conditions = _string_list(data.get("uncertain_conditions"))
    for item in explicit_confirmed:
        if item not in confirmed_conditions:
            confirmed_conditions.insert(0, item)
    for item in explicit_uncertain:
        if item not in uncertain_conditions:
            uncertain_conditions.insert(0, item)

    return {
        "raw_project_context": _clean_text(project_context, 1600),
        "confirmed_conditions": confirmed_conditions[:12],
        "uncertain_conditions": uncertain_conditions[:12],
        "ifc_facts": _string_list(data.get("ifc_facts")),
        "ifc_inferences": _string_list(data.get("ifc_inferences")),
        "conflicts": _string_list(data.get("conflicts")),
        "blocking_conflicts": _string_list(data.get("blocking_conflicts")),
        "unresolved_conditions": _string_list(data.get("unresolved_conditions")),
        "review_questions": questions,
    }


def project_preparation_prompt() -> str:
    return (
        "你是建筑消防审查项目画像整理器。只能整理输入中明确存在的信息，不得补写项目事实或规范结论。"
        "事实来源优先级必须是：用户最新明确确认的项目条件 > IFC明确字段或可复算数值 > IFC名称和用途推断。"
        "用户使用肯定语气写出的条件属于confirmed_conditions；包含可能、疑似、暂定、待确认、未知等表达的内容属于uncertain_conditions。"
        "当用户明确条件与IFC低置信度推断冲突时，以用户条件为当前审查前提，并在conflicts说明被覆盖的推断。"
        "用户报告的明确测量值与IFC明确测量值不一致时，写入blocking_conflicts；只有用户明确说明其中一个值已作废或已纠正时才可消除阻断。"
        "判定数值冲突前必须确认两者属于同一IFC实体类型、同一对象、同一指标和同一部位；不得仅因单位相同或名称相似就比较，也不得让对象名称改变其实体类型。"
        "根据当前真实可用字段生成3到6个优先复核问题，不使用固定检查清单，不重复同类构件，多个同类构件应聚合成一个问题。"
        "问题应围绕当前项目可判断的风险或缺失条件，不得继续沿用已被用户确认条件消除的不确定性。"
        "不要给出条文号、限值或合规结论。输出严格JSON："
        '{"confirmed_conditions":[],"uncertain_conditions":[],"ifc_facts":[],"ifc_inferences":[],"conflicts":[],"blocking_conflicts":[],"unresolved_conditions":[],"review_questions":[{"title":"","question":"","reason":"","facts_used":[]}]}'
    )


def compose_compliance_context(
    question: str,
    project_context: str,
    model_facts: dict[str, Any],
    project_profile: dict[str, Any] | None = None,
) -> str:
    profile = normalize_project_profile(project_profile, project_context)
    return "\n".join(
        [
            f"当前审查问题：{_clean_text(question, 2000)}",
            "事实来源优先级：用户最新明确确认的项目条件 > IFC明确字段或可复算数值 > IFC名称、用途等低置信度推断。",
            f"用户最新补充条件：{_clean_text(project_context, 1600) or '未补充'}",
            f"已确认项目条件：{json.dumps(profile['confirmed_conditions'], ensure_ascii=False)}",
            f"仍属不确定的用户条件：{json.dumps(profile['uncertain_conditions'], ensure_ascii=False)}",
            f"IFC模型事实：{json.dumps(model_facts, ensure_ascii=False)}",
            f"IFC推断与冲突说明：{json.dumps({'inferences': profile['ifc_inferences'], 'conflicts': profile['conflicts'], 'blocking_conflicts': profile['blocking_conflicts']}, ensure_ascii=False)}",
            "审查问题中出现的疑似、未确认或待确认描述可能来自较早的系统建议。若用户最新补充已明确同一事实，必须采用最新用户条件，不得再把该事实列为缺失。",
            "只有模型事实、用户条件和召回规范共同支持时才能给出合规结论；不得把IFC命名推断当成确定事实，也不得从层数擅自推导未提供的建筑高度分类条件。",
        ]
    )


def apply_project_conflict_guard(answer: dict[str, Any], project_profile: dict[str, Any]) -> dict[str, Any]:
    blocking_conflicts = _string_list(project_profile.get("blocking_conflicts"))
    if not blocking_conflicts:
        return answer
    answer["verdict"] = "INSUFFICIENT_INFORMATION"
    answer["can_answer"] = False
    answer["certainty"] = "low"
    answer["critical_risk"] = False
    answer["verdict_adjusted_by"] = "project_fact_conflict_guard"
    answer["conclusion"] = "项目条件与IFC明确事实存在尚未解决的冲突，暂不能给出通过或不通过结论。"
    missing_fields = answer.setdefault("missing_fields", [])
    for conflict in blocking_conflicts:
        field = f"冲突待确认：{conflict}"
        if field not in missing_fields:
            missing_fields.append(field)
    reasons = answer.setdefault("compliance_reasons", [])
    reasons.append("项目事实冲突门控：" + "；".join(blocking_conflicts))
    return answer
