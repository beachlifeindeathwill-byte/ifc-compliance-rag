from __future__ import annotations

import re
from typing import Any

from answer_consistency import normalize_standard_id


NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*(sqm|m2|m²|㎡|mm|cm|m|米|h|小时|min|分钟|层|人|%|倍)?",
    re.I,
)
# Require the Chinese article delimiters so decimal measurements such as 0.90m
# are never mistaken for article numbers.
ARTICLE_RE = re.compile(r"第\s*(\d+(?:\.\d+){1,3}[A-Z]?)\s*条", re.I)
STANDARD_RE = re.compile(r"GB\s*\d{5}(?:\s*-\s*\d{4})?", re.I)
PROJECT_FACT_KINDS = {"project_fact", "calculation"}


def _compact(value: Any) -> str:
    text = re.sub(r"\s+", "", str(value or "")).lower()
    return text.replace("m²", "sqm").replace("㎡", "sqm").replace("m2", "sqm")


def _citation_ids(values: Any, candidate_count: int) -> tuple[list[int], list[int]]:
    valid: list[int] = []
    invalid: list[int] = []
    for value in values if isinstance(values, list) else []:
        if not isinstance(value, int) or not 1 <= value <= candidate_count:
            if isinstance(value, int):
                invalid.append(value)
            continue
        if value not in valid:
            valid.append(value)
    return valid, invalid


def _numeric_anchors(text: str) -> list[str]:
    anchors = []
    for match in NUMBER_RE.finditer(text or ""):
        value, unit = match.groups()
        # Article numbers and PDF pages are checked by dedicated metadata rules.
        if not unit and "." in value:
            continue
        anchor = _compact(value + (unit or ""))
        if anchor not in anchors:
            anchors.append(anchor)
    return anchors


def _anchor_supported(anchor: str, evidence_text: str, grounding: str) -> bool:
    if anchor in evidence_text or anchor in grounding:
        return True
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(sqm|mm|cm|m|米|h|小时|min|分钟|层|人|%|倍)?", anchor)
    if not match:
        return False
    value, unit = match.groups()
    expected = float(value)
    for source in (evidence_text, grounding):
        for candidate in NUMBER_RE.finditer(source):
            candidate_value, candidate_unit = candidate.groups()
            normalized_unit = _compact(candidate_unit or "")
            if abs(float(candidate_value) - expected) > max(1e-9, abs(expected) * 1e-9):
                continue
            if not unit or normalized_unit == unit:
                return True
    # In extracted tables the unit commonly lives in the column header instead of
    # beside every cell. Require both the value and unit to occur in the same evidence.
    numeric_variants = {value, str(float(value)).rstrip("0").rstrip(".")}
    return bool(unit and any(item in evidence_text for item in numeric_variants) and unit in evidence_text)


def _derived_calculation_anchors(text: str, evidence_text: str, grounding: str) -> set[str]:
    derived: set[str] = set()
    forward_pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*(sqm|m2|m²|㎡|mm|cm|m|米|h|小时|min|分钟|层|人|%|倍)?"
        r"\s*[×x*]\s*(\d+(?:\.\d+)?)\s*(?:倍)?\s*=\s*"
        r"(\d+(?:\.\d+)?)\s*(sqm|m2|m²|㎡|mm|cm|m|米|h|小时|min|分钟|层|人|%)?",
        re.I,
    )
    reverse_pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*(sqm|m2|m²|㎡|mm|cm|m|米|h|小时|min|分钟|层|人|%)?"
        r"\s*[（(]\s*(\d+(?:\.\d+)?)\s*(sqm|m2|m²|㎡|mm|cm|m|米|h|小时|min|分钟|层|人|%|倍)?"
        r"\s*[×x*]\s*(\d+(?:\.\d+)?)\s*(?:倍)?\s*[）)]",
        re.I,
    )

    calculations = []
    for match in forward_pattern.finditer(text or ""):
        left_value, left_unit, multiplier, result_value, result_unit = match.groups()
        calculations.append((left_value, left_unit, multiplier, result_value, result_unit))
    for match in reverse_pattern.finditer(text or ""):
        result_value, result_unit, left_value, left_unit, multiplier = match.groups()
        calculations.append((left_value, left_unit, multiplier, result_value, result_unit))

    for left_value, left_unit, multiplier, result_value, result_unit in calculations:
        expected = float(left_value) * float(multiplier)
        if abs(expected - float(result_value)) > max(1e-6, abs(expected) * 1e-6):
            continue
        left_anchor = _compact(left_value + (left_unit or ""))
        multiplier_anchor = _compact(multiplier + "倍")
        multiplier_supported = (
            _anchor_supported(multiplier_anchor, evidence_text, grounding)
            or multiplier in evidence_text
            or multiplier in grounding
        )
        if _anchor_supported(left_anchor, evidence_text, grounding) and multiplier_supported:
            derived.add(_compact(result_value + (result_unit or left_unit or "")))
    return derived


def _claim_evidence_text(citations: list[int], candidates: list[dict[str, Any]]) -> str:
    parts = []
    for citation in citations:
        candidate = candidates[citation - 1]
        parts.extend(
            str(candidate.get(field) or "")
            for field in ("standard_id", "article_no", "table_no", "page", "text")
        )
    return _compact(" ".join(parts))


def _grounding_with_typed_numeric_aliases(grounding_text: str) -> str:
    """Expand serialized model fields so typed values can be checked as measurements."""
    compacted = _compact(grounding_text)
    aliases = [compacted]
    unit_suffixes = {
        "_m": "m",
        "_mm": "mm",
        "_cm": "cm",
        "_m2": "sqm",
        "_sqm": "sqm",
        "_h": "h",
    }
    field_pattern = re.compile(r'["\']?([\w\u4e00-\u9fff]+(?:_m2|_sqm|_mm|_cm|_m|_h))["\']?[:：](\d+(?:\.\d+)?)')
    for match in field_pattern.finditer(compacted):
        field, value = match.groups()
        suffix = next((item for item in unit_suffixes if field.endswith(item)), "")
        if suffix:
            aliases.append(value + unit_suffixes[suffix])
    return " ".join(aliases)


def audit_atomic_claims(
    answer: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    grounding_text: str = "",
) -> dict[str, Any]:
    """Audit explicit anchors for each model-produced atomic claim.

    This deliberately avoids semantic keyword heuristics. It checks only facts that can be
    verified deterministically: citation bounds, cited standards/articles and numeric anchors.
    """
    raw_claims = answer.get("atomic_claims")
    if not isinstance(raw_claims, list):
        raw_claims = []

    rows: list[dict[str, Any]] = []
    all_issues: list[str] = []
    grounding = _grounding_with_typed_numeric_aliases(grounding_text)
    for index, raw_claim in enumerate(raw_claims, start=1):
        if not isinstance(raw_claim, dict):
            issue = f"原子结论 {index} 的结构无效"
            rows.append({"index": index, "passed": False, "issues": [issue]})
            all_issues.append(issue)
            continue

        text = str(raw_claim.get("text") or "").strip()
        kind = str(raw_claim.get("kind") or "requirement").strip().lower()
        citations, invalid = _citation_ids(raw_claim.get("citations"), len(candidates))
        issues: list[str] = []
        if not text:
            issues.append("结论文本为空")
        if invalid:
            issues.append("引用编号越界：" + "、".join(map(str, invalid)))
        if kind not in PROJECT_FACT_KINDS and not citations:
            issues.append("规范结论没有绑定引用")

        evidence_text = _claim_evidence_text(citations, candidates)
        derived_anchors = _derived_calculation_anchors(text, evidence_text, grounding) if kind == "calculation" else set()
        for standard in STANDARD_RE.findall(text):
            if _compact(normalize_standard_id(standard)) not in evidence_text:
                issues.append(f"引用证据不包含标准 {normalize_standard_id(standard)}")
        for article in ARTICLE_RE.findall(text):
            if _compact(article) not in evidence_text:
                issues.append(f"引用证据不包含条文 {article}")
        for anchor in _numeric_anchors(text):
            if not _anchor_supported(anchor, evidence_text, grounding) and anchor not in derived_anchors:
                issues.append(f"数值锚点 {anchor} 未出现在引用证据或用户事实中")

        issues = list(dict.fromkeys(issues))
        all_issues.extend(f"原子结论 {index}：{issue}" for issue in issues)
        rows.append(
            {
                "index": index,
                "text": text,
                "kind": kind,
                "citations": citations,
                "passed": not issues,
                "issues": issues,
            }
        )

    return {
        "claims_present": bool(raw_claims),
        "claims": rows,
        "claim_count": len(rows),
        "supported_claim_count": sum(1 for row in rows if row.get("passed")),
        "unsupported_claim_count": sum(1 for row in rows if not row.get("passed")),
        "passed": bool(rows) and not all_issues,
        "issues": all_issues,
    }


def answer_quality_snapshot(
    answer: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    grounding_text: str = "",
) -> dict[str, Any]:
    citations, invalid = _citation_ids(answer.get("citations"), len(candidates))
    claim_audit = audit_atomic_claims(answer, candidates, grounding_text=grounding_text)
    structural_issues = []
    if answer.get("can_answer") and not citations:
        structural_issues.append("可回答结论缺少有效引用")
    if invalid:
        structural_issues.append("回答引用编号越界：" + "、".join(map(str, invalid)))
    if answer.get("certainty") not in {"high", "medium", "low"}:
        structural_issues.append("置信度字段无效")
    return {
        "passed": not structural_issues and (not claim_audit["claims_present"] or claim_audit["passed"]),
        "valid_citations": citations,
        "structural_issues": structural_issues,
        "claim_audit": claim_audit,
    }
