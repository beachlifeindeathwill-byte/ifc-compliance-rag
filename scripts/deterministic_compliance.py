from __future__ import annotations

import math
import re
from typing import Any


SAFETY_METRIC_TERMS = (
    "疏散",
    "安全出口",
    "防火",
    "耐火",
    "防火分区",
    "面积",
    "距离",
    "净宽",
)

UNIT_ALIASES = {
    "m": "m",
    "米": "m",
    "meters": "m",
    "meter": "m",
    "cm": "cm",
    "厘米": "cm",
    "mm": "mm",
    "毫米": "mm",
    "m2": "m2",
    "m²": "m2",
    "㎡": "m2",
    "平方米": "m2",
    "平方": "m2",
    "sqm": "m2",
    "h": "h",
    "hr": "h",
    "hrs": "h",
    "hour": "h",
    "hours": "h",
    "小时": "h",
    "min": "min",
    "minutes": "min",
    "分钟": "min",
    "%": "percent",
    "percent": "percent",
}

UNIT_CONVERSION_TO_BASE = {
    "m": ("m", 1.0),
    "cm": ("m", 0.01),
    "mm": ("m", 0.001),
    "m2": ("m2", 1.0),
    "h": ("h", 1.0),
    "min": ("h", 1.0 / 60.0),
    "percent": ("percent", 1.0),
}

COMPARATOR_ALIASES = {
    "lte": "lte",
    "le": "lte",
    "<=": "lte",
    "≤": "lte",
    "不超过": "lte",
    "不应大于": "lte",
    "不大于": "lte",
    "lt": "lt",
    "<": "lt",
    "小于": "lt",
    "gte": "gte",
    "ge": "gte",
    ">=": "gte",
    "≥": "gte",
    "不低于": "gte",
    "不应小于": "gte",
    "不小于": "gte",
    "gt": "gt",
    ">": "gt",
    "大于": "gt",
    "eq": "eq",
    "=": "eq",
    "等于": "eq",
}


def _normalize_unit(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().lower().replace(" ", "")
    if not raw:
        return None
    return UNIT_ALIASES.get(raw, raw)


def _parse_number(value: Any) -> tuple[float | None, str | None]:
    if isinstance(value, bool):
        return None, None
    if isinstance(value, (int, float)):
        return float(value), None

    raw = str(value or "").strip().replace(",", "")
    if not raw:
        return None, None

    unit = None
    if "%" in raw:
        unit = "percent"
        raw = raw.replace("%", "")

    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", raw)
    if not match:
        return None, unit
    try:
        return float(match.group(0)), unit
    except ValueError:
        return None, unit


def _to_base(value: float, unit: str | None) -> tuple[float | None, str | None]:
    normalized = _normalize_unit(unit)
    if normalized is None:
        return None, None
    conversion = UNIT_CONVERSION_TO_BASE.get(normalized)
    if conversion is None:
        return value, normalized
    base_unit, factor = conversion
    return value * factor, base_unit


def _normalize_comparator(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().lower().replace(" ", "")
    if not raw:
        return None
    if raw.startswith("actual"):
        raw = raw.replace("actual", "", 1)
    return COMPARATOR_ALIASES.get(raw)


def _is_compliant(actual: float, requirement: float, comparator: str) -> bool:
    epsilon = 1e-9 * max(1.0, abs(actual), abs(requirement))
    if comparator == "lte":
        return actual <= requirement + epsilon
    if comparator == "lt":
        return actual < requirement - epsilon
    if comparator == "gte":
        return actual >= requirement - epsilon
    if comparator == "gt":
        return actual > requirement + epsilon
    if comparator == "eq":
        return math.isclose(actual, requirement, rel_tol=1e-9, abs_tol=epsilon)
    return False


def _number_matches_source(value: float, unit: str | None, source: str) -> bool:
    """Check that the extracted number is present in a source text, not only model output."""
    if not source:
        return False
    text = str(source)
    normalized_unit = _normalize_unit(unit)
    unit_patterns = []
    if normalized_unit == "m":
        unit_patterns = ["m", "米"]
    elif normalized_unit == "m2":
        unit_patterns = ["㎡", "m²", "m2", "平方米", "平方"]
    elif normalized_unit == "h":
        unit_patterns = ["h", "小时"]
    elif normalized_unit == "percent":
        unit_patterns = ["%", "percent"]
    elif normalized_unit:
        unit_patterns = [normalized_unit]

    if value.is_integer():
        number_variants = {str(int(value)), f"{value:g}", f"{value:.0f}"}
    else:
        number_variants = {f"{value:g}", f"{value:.10g}"}

    for number_text in number_variants:
        if unit_patterns:
            for unit_text in unit_patterns:
                trailing_zeros = r"(?:\.0*)?" if "." not in number_text else r"0*"
                if re.search(
                    re.escape(number_text) + trailing_zeros + r"\s*" + re.escape(unit_text),
                    text,
                    flags=re.IGNORECASE,
                ):
                    return True
        elif number_text in text:
            return True
    return False


def _check_sources_match(check: dict[str, Any], context: list[str]) -> bool:
    context_text = "\n".join(context)
    if not context_text:
        return False
    actual, actual_unit = _parse_number(check.get("actual_value"))
    requirement, requirement_unit = _parse_number(check.get("requirement_value"))
    if actual is None or requirement is None:
        return False
    actual_source_matched = _number_matches_source(
        actual,
        check.get("actual_unit") or actual_unit,
        context_text,
    )
    requirement_source_matched = _number_matches_source(
        requirement,
        check.get("requirement_unit") or requirement_unit,
        context_text,
    )
    return actual_source_matched and requirement_source_matched


def evaluate_numeric_check(check: dict[str, Any], context: list[str] | None = None) -> dict[str, Any]:
    actual, embedded_actual_unit = _parse_number(check.get("actual_value"))
    requirement, embedded_requirement_unit = _parse_number(check.get("requirement_value"))
    actual_unit = check.get("actual_unit") or embedded_actual_unit
    requirement_unit = check.get("requirement_unit") or embedded_requirement_unit
    comparator = _normalize_comparator(check.get("comparator"))

    base: dict[str, Any] = {
        "metric": str(check.get("metric") or "").strip(),
        "citation": check.get("citation"),
        "actual_value": actual,
        "requirement_value": requirement,
        "actual_unit": actual_unit,
        "requirement_unit": requirement_unit,
        "comparator": comparator,
        "status": "verified",
        "actual_result": None,
        "expected_result": str(check.get("expected_result") or "").strip().upper(),
        "source_matched": False,
        "reason": "",
    }

    if actual is None or requirement is None:
        base.update(
            status="missing_value",
            reason="实际值或规范限值缺失，无法完成确定性比较。",
        )
        return base
    if comparator is None:
        base.update(
            status="invalid_comparator",
            reason="比较方向无法识别，需要人工确认。",
        )
        return base

    actual_base, actual_base_unit = _to_base(actual, actual_unit)
    requirement_base, requirement_base_unit = _to_base(requirement, requirement_unit)
    if actual_base is None or requirement_base is None or actual_base_unit != requirement_base_unit:
        base.update(
            status="unit_mismatch",
            reason="实际值与规范限值单位不兼容，无法自动比较。",
        )
        return base

    compliant = _is_compliant(actual_base, requirement_base, comparator)
    base["actual_result"] = "PASS" if compliant else "FAIL"
    base["source_matched"] = _check_sources_match(check, context or [])
    if not base["source_matched"]:
        base["reason"] = "数值比较已完成，但提取值未在传入检索文本中同时找到，自动修正 PASS 时仍需复核。"
    return base


def apply_numeric_compliance_guard(
    answer: dict[str, Any],
    *,
    retrieval_question: str = "",
    candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    checks = answer.get("numeric_checks")
    if not isinstance(checks, list) or not checks:
        return answer

    candidates = candidates or []
    context: list[str] = []
    if retrieval_question:
        context.append(retrieval_question)
    cited = {int(item) for item in (answer.get("citations") or []) if isinstance(item, int)}
    for index, candidate in enumerate(candidates, start=1):
        if index in cited:
            context.append(str(candidate.get("text") or ""))

    evaluations = [evaluate_numeric_check(check, context) for check in checks]
    answer["numeric_verification"] = evaluations
    verification_reasons: list[str] = []
    for item in evaluations:
        if item["status"] == "verified":
            verification_reasons.append(
                f"确定性数值复核：{item['metric']} {item['actual_value']}"
                f"{item['actual_unit'] or ''} 与限值 {item['requirement_value']}"
                f"{item['requirement_unit'] or ''}，判定 {item['actual_result']}。"
            )
    if verification_reasons:
        answer.setdefault("compliance_reasons", [])
        answer["compliance_reasons"] = [
            *answer["compliance_reasons"],
            *verification_reasons,
        ]

    original_verdict = str(answer.get("verdict") or "INSUFFICIENT_INFORMATION").strip().upper()
    new_verdict = original_verdict
    verified_failures = [item for item in evaluations if item["actual_result"] == "FAIL"]
    unverifiable = [
        item
        for item in evaluations
        if item["status"] in {"missing_value", "invalid_comparator", "unit_mismatch"}
    ]
    verified_checks = [
        item for item in evaluations if item["status"] == "verified"
    ]
    all_pass = bool(verified_checks) and all(
        item["actual_result"] == "PASS" for item in verified_checks
    )
    all_sources_matched = bool(verified_checks) and all(
        item["source_matched"] for item in verified_checks
    )
    decision_basis = str(answer.get("decision_basis") or "").strip().lower()
    if decision_basis not in {"numeric_only", "mixed"}:
        non_numeric_failures = answer.get("non_numeric_failures")
        decision_basis = (
            "numeric_only"
            if isinstance(non_numeric_failures, list)
            and not non_numeric_failures
            and not answer.get("missing_fields")
            else "mixed"
        )
    explicit_blocking_conditions = answer.get("blocking_conditions")
    non_numeric_failures = (
        explicit_blocking_conditions
        if isinstance(explicit_blocking_conditions, list)
        else answer.get("non_numeric_failures")
    )
    blocking_non_numeric = (
        decision_basis == "mixed"
        and isinstance(non_numeric_failures, list)
        and bool(non_numeric_failures)
        and not answer.get("unsupported_claims_excluded")
    )

    if verified_failures:
        new_verdict = "FAIL"
    elif new_verdict == "PASS" and blocking_non_numeric:
        new_verdict = "INSUFFICIENT_INFORMATION"
    elif new_verdict == "PASS" and unverifiable:
        new_verdict = "INSUFFICIENT_INFORMATION"
    elif (
        new_verdict in {"FAIL", "INSUFFICIENT_INFORMATION"}
        and decision_basis == "numeric_only"
        and all_pass
        and all_sources_matched
    ):
        new_verdict = "PASS"

    if new_verdict != original_verdict:
        answer["verdict"] = new_verdict
        answer["verdict_adjusted_by"] = (
            "non_numeric_condition_guard"
            if blocking_non_numeric and new_verdict == "INSUFFICIENT_INFORMATION"
            else "deterministic_numeric_check"
        )
        if new_verdict == "FAIL":
            answer["can_answer"] = True
            answer["certainty"] = "medium"
            answer["conclusion"] = (
                "数值规则复核发现实际值不满足规范限值；请核对原始条文、模型字段和适用条件后确认。"
            )
            if any(
                term in str(item.get("metric") or "")
                for item in verified_failures
                for term in SAFETY_METRIC_TERMS
            ):
                answer["critical_risk"] = True
            answer.setdefault("missing_fields", [])
            field = "数值复核对应的模型实际值、规范限值和条文来源"
            if field not in answer["missing_fields"]:
                answer["missing_fields"].append(field)
            reason = str(answer.get("confidence_reason") or "")
            suffix = "确定性数值比较已纠正模型算术判断。"
            answer["confidence_reason"] = f"{reason} {suffix}".strip()
        elif new_verdict == "PASS":
            answer["can_answer"] = True
            answer["certainty"] = "medium"
            answer["conclusion"] = (
                "数值规则复核后，系统提取的实际值满足规范限值；仍需人工核对适用条件和字段来源。"
            )
            answer.setdefault("missing_fields", [])
            field = "数值提取与适用条文的对应关系"
            if field not in answer["missing_fields"]:
                answer["missing_fields"].append(field)
            reason = str(answer.get("confidence_reason") or "")
            suffix = "确定性数值比较已纠正模型比较结果。"
            answer["confidence_reason"] = f"{reason} {suffix}".strip()
        else:
            answer["can_answer"] = False
            answer["certainty"] = "low"
            answer["conclusion"] = (
                "数值比较本身可以完成，但仍有影响规范适用性的项目条件未确认，暂不能给出通过或不通过结论。"
                if blocking_non_numeric
                else "数值规则引擎无法完成可靠比较，需人工确认实际值与规范限值的对应关系。"
            )
            answer.setdefault("missing_fields", [])
            field = (
                "影响规范适用性的非数值项目条件"
                if blocking_non_numeric
                else "可参与确定性比较的实际值和规范限值"
            )
            if field not in answer["missing_fields"]:
                answer["missing_fields"].append(field)

    return answer
