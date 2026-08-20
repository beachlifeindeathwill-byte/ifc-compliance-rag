from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryIntent:
    article_refs: list[str]
    page_refs: list[int]
    table_refs: list[str]
    asks_version_compare: bool
    asks_locator: bool
    lacks_project_context: bool
    asks_local_rule: bool


ARTICLE_RE = re.compile(r"(?:第\s*)?([1-9][0-9]*(?:\.[0-9A-Za-z]+){1,3})\s*条?")
PAGE_RE = re.compile(r"(?:第\s*)?([0-9]{1,4})\s*页")
TABLE_RE = re.compile(r"(?:表|续表)\s*([0-9]+(?:\.[0-9]+)+(?:-\d+)?)")
STANDARD_RE = re.compile(r"GB\s*\d{5}\s*[-－]\s*\d{4}|\d{4}\s*年版|\d{4}\s*版")


def _dedupe(items: list) -> list:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def detect_query_intent(question: str) -> QueryIntent:
    text = question or ""
    article_refs = _dedupe([match.group(1) for match in ARTICLE_RE.finditer(text)])
    page_refs = _dedupe([int(match.group(1)) for match in PAGE_RE.finditer(text)])
    table_refs = _dedupe([match.group(1) for match in TABLE_RE.finditer(text)])

    compare_terms = ["区别", "差异", "不同", "变化", "修订", "更新", "对比", "版本"]
    version_hits = STANDARD_RE.findall(text)
    asks_version_compare = any(term in text for term in compare_terms) and (
        len(version_hits) >= 1 or "年版" in text or "版本" in text
    )

    vague_design_terms = ["合规吗", "是否合规", "这样设计", "这个建筑", "我的建筑", "这个方案"]
    detail_terms = [
        "建筑类型",
        "建筑高度",
        "耐火等级",
        "防火分区",
        "疏散",
        "汽车库",
        "厂房",
        "仓库",
        "住宅",
        "公共建筑",
        "中庭",
        "面积",
        "净宽",
        "距离",
    ]
    negated_context_terms = any(
        marker in text and term in text
        for marker in ["没有", "不知", "不知道", "不清楚", "未提供", "未上传", "缺少"]
        for term in ["IFC", "建筑类型", "建筑高度", "层数", "面积", "尺寸", "用途", "项目条件"]
    )
    asks_whole_building_judgement = any(term in text for term in vague_design_terms)
    lacks_project_context = negated_context_terms or (
        asks_whole_building_judgement and not any(term in text for term in detail_terms)
    )
    asks_local_rule = any(term in text for term in ["地方标准", "地方消防", "地方审查", "审查口径", "南京"])
    asks_locator = bool(article_refs or page_refs or table_refs)

    return QueryIntent(
        article_refs=article_refs,
        page_refs=page_refs,
        table_refs=table_refs,
        asks_version_compare=asks_version_compare,
        asks_locator=asks_locator,
        lacks_project_context=lacks_project_context,
        asks_local_rule=asks_local_rule,
    )
