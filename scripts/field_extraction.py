from __future__ import annotations

import re


NUMBER_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:m|米|h|㎡|m2|m²|%|人|层|个|辆|倍)",
    re.IGNORECASE,
)

LEVEL_RE = re.compile(r"(?:甲级|乙级|丙级|一[、,，]二级|[一二三四ⅠⅡⅢⅣIVX]+类)")

REQUIREMENT_TERMS = [
    "不应小于",
    "不应大于",
    "不应少于",
    "不宜小于",
    "不宜大于",
    "应设置",
    "可设置",
    "应符合",
    "应采用",
    "严禁",
    "不得",
    "耐火极限",
    "使用面积",
    "短边",
    "净宽度",
    "疏散距离",
    "防火分隔",
    "安全出口",
    "自动灭火系统",
]

CONDITION_TERMS = [
    "当",
    "除",
    "时",
    "若",
    "位于",
    "设置",
    "未设置",
    "地下",
    "半地下",
    "首层",
    "高层",
    "公共建筑",
    "住宅建筑",
    "汽车库",
]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_evidence_fields(text: str) -> dict:
    compact = clean_text(text)
    numbers = [match.group(0).replace(" ", "") for match in NUMBER_RE.finditer(compact)]
    levels = [match.group(0) for match in LEVEL_RE.finditer(compact)]
    requirements = [term for term in REQUIREMENT_TERMS if term in compact]
    conditions = [term for term in CONDITION_TERMS if term in compact]
    return {
        "numbers": list(dict.fromkeys(numbers)),
        "levels": list(dict.fromkeys(levels)),
        "requirements": list(dict.fromkeys(requirements)),
        "conditions": list(dict.fromkeys(conditions)),
        "has_key_field": bool(numbers or levels or requirements),
    }


def field_recall_status(item: dict | None) -> list[tuple[str, bool, str]]:
    if not item:
        return [
            ("标准名称", False, "未召回"),
            ("条文号", False, "未召回"),
            ("页码", False, "未召回"),
            ("原文片段", False, "未召回"),
            ("数值或关键字段", False, "未召回"),
        ]

    text = item.get("text", "") or ""
    fields = extract_evidence_fields(text)
    key_values = fields["numbers"][:3] or fields["levels"][:3] or fields["requirements"][:3]
    key_value_text = "、".join(key_values) if key_values else "未直接召回"
    page = item.get("page")
    page_text = f"PDF物理页第 {page} 页" if page is not None else "未召回"
    return [
        ("标准名称", bool(item.get("standard_id")), item.get("standard_id") or "未召回"),
        ("条文号", bool(item.get("article_no")), item.get("article_no") or "未召回"),
        ("页码", page is not None, page_text),
        ("原文片段", bool(clean_text(text)), "已召回" if clean_text(text) else "未召回"),
        ("数值或关键字段", fields["has_key_field"], key_value_text),
    ]
