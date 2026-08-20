from __future__ import annotations

import re
from typing import Any


STANDARD_RE = re.compile(r"GB\s*\d{5}(?:\s*-\s*\d{4})?", re.I)
ARTICLE_RE = re.compile(
    r"(GB\s*\d{5}(?:\s*-\s*\d{4})?).{0,40}?第?\s*(\d+(?:\.\d+){1,3}[A-Z]?)\s*条?",
    re.I,
)
COMPARISON_RE = re.compile(
    r"(不应大于|不应小于|不大于|不小于|大于|小于|超过|不超过|等于|达到)"
    r"\s*(\d+(?:\.\d+)?)\s*(m2|m²|㎡|mm|cm|m|米|h|小时|min|分钟|层|人|%)?",
    re.I,
)
CAUSAL_FACT_RE = re.compile(
    r"(?:因|由于|基于|已知|模型显示|实际)"
    r"(?P<subject>[\u4e00-\u9fffA-Za-z0-9_、，]{2,24}?)"
    r"(?P<comparator>大于|小于|超过|不超过|达到|等于)"
    r"(?P<value>\d+(?:\.\d+)?)"
    r"(?P<unit>m2|m²|㎡|mm|cm|m|米|h|小时|min|分钟|层|人|%)?",
    re.I,
)
UNCERTAINTY_TERMS = ("是否", "未明确", "未知", "缺少", "待确认", "需确认", "需要确认", "尚未确认")
TEXT_FIELDS = ("conclusion", "basis", "confidence_reason", "note")


def normalize_standard_id(value: str) -> str:
    compact = re.sub(r"\s+", "", str(value or "")).upper()
    match = re.fullmatch(r"GB(\d{5})(?:-(\d{4}))?", compact)
    if not match:
        return compact
    return f"GB {match.group(1)}" + (f"-{match.group(2)}" if match.group(2) else "")


def _answer_text(answer: dict[str, Any]) -> str:
    parts = [str(answer.get(field) or "") for field in TEXT_FIELDS]
    parts.extend(str(item) for item in answer.get("compliance_reasons") or [])
    return " ".join(parts)


def _mentioned_standard_articles(text: str) -> list[tuple[str, str | None]]:
    article_matches = [
        (normalize_standard_id(match.group(1)), match.group(2))
        for match in ARTICLE_RE.finditer(text)
    ]
    standards_with_articles = {standard for standard, _ in article_matches}
    standard_only = [
        (normalize_standard_id(match.group(0)), None)
        for match in STANDARD_RE.finditer(text)
        if normalize_standard_id(match.group(0)) not in standards_with_articles
    ]
    return list(dict.fromkeys(article_matches + standard_only))


def _article_matches(candidate_article: str | None, mentioned_article: str | None) -> bool:
    if not mentioned_article:
        return True
    candidate_article = str(candidate_article or "")
    return (
        candidate_article == mentioned_article
        or candidate_article.startswith(mentioned_article + ".")
        or mentioned_article.startswith(candidate_article + ".")
    )


def ensure_mentioned_evidence_citations(
    answer: dict[str, Any], candidates: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[str]]:
    """Attach available evidence for every standard/article named by the answer."""
    issues: list[str] = []
    citations = [
        int(item)
        for item in answer.get("citations") or []
        if isinstance(item, int) and 1 <= item <= len(candidates)
    ]
    cited = set(citations)

    for standard_id, article_no in _mentioned_standard_articles(_answer_text(answer)):
        matching = [
            index
            for index, candidate in enumerate(candidates, start=1)
            if normalize_standard_id(candidate.get("standard_id", "")) == standard_id
            and _article_matches(candidate.get("article_no"), article_no)
        ]
        if not matching and article_no:
            matching = [
                index
                for index, candidate in enumerate(candidates, start=1)
                if normalize_standard_id(candidate.get("standard_id", "")) == standard_id
            ]
        if matching:
            if not any(index in cited for index in matching):
                citations.append(matching[0])
                cited.add(matching[0])
        else:
            label = f"{standard_id} 第{article_no}条" if article_no else standard_id
            issues.append(f"回答提到的 {label} 不在本次召回证据中")

    answer["citations"] = list(dict.fromkeys(citations))
    return answer, issues


def _missing_subject(text: str, comparison_start: int) -> str:
    prefix = text[:comparison_start]
    for term in UNCERTAINTY_TERMS:
        prefix = prefix.replace(term, "")
    prefix = re.sub(r"[（(].*$", "", prefix)
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", prefix)
    return chinese_terms[-1][-12:] if chinese_terms else ""


def _is_conditional_occurrence(text: str, start: int, end: int) -> bool:
    prefix = text[max(0, start - 12) : start]
    suffix = text[end : end + 8]
    if any(term in prefix for term in ("若", "如果", "当", "在")):
        return True
    return suffix.startswith(("时", "则", "的情况下", "条件下"))


def missing_field_contradictions(answer: dict[str, Any]) -> list[str]:
    """Find project facts asserted despite being declared missing by the answer."""
    conclusion = re.sub(r"\s+", "", str(answer.get("conclusion") or ""))
    issues: list[str] = []
    for raw_field in answer.get("missing_fields") or []:
        field = re.sub(r"\s+", "", str(raw_field or ""))
        if not field or not any(term in field for term in UNCERTAINTY_TERMS):
            continue
        for match in COMPARISON_RE.finditer(field):
            predicate = "".join(part or "" for part in match.groups())
            subject = _missing_subject(field, match.start())
            occurrences = list(re.finditer(re.escape(predicate), conclusion))
            asserted = False
            for occurrence in occurrences:
                if _is_conditional_occurrence(conclusion, occurrence.start(), occurrence.end()):
                    continue
                context_start = max(0, occurrence.start() - max(16, len(subject) + 6))
                context = conclusion[context_start : occurrence.start()]
                if subject and subject not in context:
                    continue
                asserted = True
                break
            if not asserted:
                continue
            issues.append(f"结论将缺失条件“{raw_field}”当作已确认事实")
    return list(dict.fromkeys(issues))


def unsupported_project_fact_assertions(answer: dict[str, Any], grounding_text: str) -> list[str]:
    """Detect causal numeric project facts that do not occur in user-provided facts."""
    conclusion = re.sub(r"\s+", "", str(answer.get("conclusion") or ""))
    grounding = re.sub(r"\s+", "", str(grounding_text or ""))
    issues: list[str] = []
    for match in CAUSAL_FACT_RE.finditer(conclusion):
        subject = match.group("subject").strip("，、,:：；;。")
        predicate = f"{subject}{match.group('comparator')}{match.group('value')}{match.group('unit') or ''}"
        if predicate in grounding:
            continue
        issues.append(f"结论使用了用户未提供的项目事实“{predicate}”")
    return list(dict.fromkeys(issues))


def apply_answer_consistency_guard(
    answer: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    grounding_text: str = "",
    allow_partial_claims: bool = False,
) -> dict[str, Any]:
    from answer_quality import answer_quality_snapshot

    answer, citation_issues = ensure_mentioned_evidence_citations(answer, candidates)
    fact_issues = missing_field_contradictions(answer)
    unsupported_issues = unsupported_project_fact_assertions(answer, grounding_text)
    for issue in unsupported_issues:
        if issue not in fact_issues:
            fact_issues.append(issue)
            missing = answer.setdefault("missing_fields", [])
            field = issue.replace("结论使用了用户未提供的项目事实", "项目事实来源")
            if field not in missing:
                missing.append(field)
    quality = answer_quality_snapshot(answer, candidates, grounding_text=grounding_text)
    answer["quality_audit"] = quality
    claim_issues = quality["claim_audit"]["issues"]
    issues = citation_issues + fact_issues + claim_issues
    partial_claim_repair = False
    if not issues:
        return answer

    answer["consistency_issues"] = issues
    answer["answer_adjusted_by"] = "post_generation_consistency_guard"
    if citation_issues:
        answer["can_answer"] = False
        answer["certainty"] = "low"
        answer["conclusion"] = "回答引用了本次证据中不存在的规范依据，无法可靠给出结论。"
        answer["confidence_reason"] = "；".join(citation_issues)
    elif claim_issues and allow_partial_claims and quality["claim_audit"]["supported_claim_count"]:
        partial_claim_repair = True
        answer["unsupported_claims_excluded"] = True
        unsupported = [
            row for row in quality["claim_audit"]["claims"] if not row.get("passed")
        ]
        answer["omitted_atomic_claims"] = unsupported
        answer["atomic_claims"] = [
            row
            for row in answer.get("atomic_claims") or []
            if isinstance(row, dict)
            and row.get("text") not in {item.get("text") for item in unsupported}
        ]
        answer["certainty"] = "medium"
        answer["confidence_reason"] = "部分补充结论未获得完整证据支持，已从合规判断依据中排除：" + "；".join(claim_issues)
        verdict = str(answer.get("verdict") or "").upper()
        if verdict == "PASS":
            answer["conclusion"] = "经已验证的规范证据和数值规则复核，已知条件满足对应限值；未获完整证据支持的补充结论未用于判断。"
        elif verdict == "FAIL":
            answer["conclusion"] = "经已验证的规范证据和数值规则复核，已知条件不满足对应限值；未获完整证据支持的补充结论未用于判断。"
        else:
            answer["conclusion"] = "部分独立结论缺少完整证据，当前只能保留已验证的条件分支。"
        existing_note = str(answer.get("note") or "").strip()
        guard_note = "系统已排除未获完整证据支持的补充结论。"
        answer["note"] = f"{existing_note} {guard_note}".strip()
    elif claim_issues:
        answer["can_answer"] = False
        answer["certainty"] = "low"
        answer["conclusion"] = "回答中的部分独立结论没有获得引用证据完整支持，无法可靠给出结论。"
        answer["confidence_reason"] = "；".join(claim_issues)
        existing_note = str(answer.get("note") or "").strip()
        guard_note = "系统已拦截缺少证据支撑的独立结论。"
        answer["note"] = f"{existing_note} {guard_note}".strip()
    elif fact_issues:
        answer["can_answer"] = True
        answer["certainty"] = "medium"
        answer["conclusion"] = "已召回到相关规范要求，但部分项目条件尚未确认，不能据此给出唯一结论；请按已列条件分支复核。"
        answer["confidence_reason"] = "；".join(fact_issues)
        existing_note = str(answer.get("note") or "").strip()
        guard_note = "系统已拦截将未确认项目条件当作已知事实的表述。"
        answer["note"] = f"{existing_note} {guard_note}".strip()

    if not partial_claim_repair and str(answer.get("verdict") or "").upper() in {"PASS", "FAIL"}:
        answer["verdict"] = "INSUFFICIENT_INFORMATION"
        answer["critical_risk"] = False
    return answer
