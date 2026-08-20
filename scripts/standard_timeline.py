from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "data" / "policy"
TIMELINE_FILE = POLICY_DIR / "standard_timeline.json"
REGISTRY_FILE = POLICY_DIR / "standard_registry.json"
ABOLISHED_FILE = POLICY_DIR / "abolished_articles_seed.json"

ROLE_LABELS = {
    "governing_residential_code": "住宅项目通用规范",
    "governing_general_code": "建筑防火通用规范",
    "governing_facility_code": "消防设施通用规范",
    "specialized_legacy_code": "汽车库专项设计规范",
    "legacy_design_code": "建筑设计防火规范（旧版设计规范）",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_standard_timeline() -> dict[str, Any]:
    return _load_json(TIMELINE_FILE)


def load_standard_registry() -> dict[str, Any]:
    return _load_json(REGISTRY_FILE)


def load_abolished_seed() -> dict[str, Any]:
    return _load_json(ABOLISHED_FILE)


def _sort_versions(versions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(versions, key=lambda item: str(item.get("effective_date", "")))


def _clean_keywords(value: Any) -> list[str]:
    return [re.sub(r"\s+", "", str(item)) for item in (value or []) if str(item).strip()]


def _slug(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_").lower()


def _registry_version(
    registry: dict[str, Any],
    standard_id: str,
    *,
    status: str,
    status_label: str,
    article_no: str = "",
    requirement: str | None = None,
    source_excerpt: str | None = None,
) -> dict[str, Any]:
    config = registry.get("standards", {}).get(standard_id) or {}
    title = config.get("title") or standard_id
    effective_date = config.get("effective_date") or ""
    role_code = config.get("role") or ""
    role = config.get("category") or ROLE_LABELS.get(role_code) or role_code or "规范"
    return {
        "standard_id": standard_id,
        "title": title,
        "article_no": article_no,
        "page": "规范注册表",
        "effective_date": effective_date,
        "status": status,
        "status_label": status_label,
        "role": role,
        "requirement": requirement or f"{title}自{effective_date or '未标注日期'}起施行，当前角色为{role}。",
        "source_excerpt": source_excerpt or f"标准注册表：{title}，施行日期 {effective_date or '未标注'}，类别 {config.get('category') or role}。",
    }


def _effective_timeline_topic(registry: dict[str, Any]) -> dict[str, Any]:
    versions: list[dict[str, Any]] = []
    for standard_id, config in registry.get("standards", {}).items():
        effective_date = config.get("effective_date") or ""
        versions.append(
            _registry_version(
                registry,
                standard_id,
                status="current",
                status_label="已收录规范",
                requirement=f"{config.get('title') or standard_id}自{effective_date or '未标注日期'}起施行，当前角色为{config.get('category') or ROLE_LABELS.get(config.get('role') or '') or '规范'}。",
                source_excerpt=f"标准注册表：{standard_id}，施行日期 {effective_date or '未标注'}，类别 {config.get('category') or ROLE_LABELS.get(config.get('role') or '') or '规范'}。",
            )
        )
    versions = _sort_versions(versions)
    first = versions[0].get("effective_date") if versions else ""
    latest = versions[-1].get("effective_date") if versions else ""
    return {
        "topic_id": "standard_effective_timeline",
        "title": "已收录规范施行时间轴",
        "subject": "当前知识库中全部规范",
        "change_summary": f"知识库当前收录 {len(versions)} 份规范，施行日期从 {first} 到 {latest}。新增规范后，本主题由标准注册表自动扩展。",
        "applicability_note": "本主题展示规范级时间轴，不直接代替条文级对比；条文差异需要结合废止关系和原文证据逐条核对。",
        "keywords": ["规范施行时间轴", "标准时间轴", "施行日期", "标准实施", "规范时间轴"],
        "versions": versions,
        "topic_kind": "generated",
        "source": "standard_registry.json",
    }


def _replacement_topic(registry: dict[str, Any], rule: dict[str, Any], source_info: dict[str, Any] | None = None) -> dict[str, Any]:
    source_id = rule.get("source_standard_id") or ""
    target_id = rule.get("replaced_by_standard_id") or ""
    source_title = rule.get("source_title") or source_id
    target_title = rule.get("replaced_by_title") or target_id
    articles = rule.get("articles") or []
    samples = articles[:12]
    return {
        "topic_id": _slug(f"replacement_{source_id}_to_{target_id}"),
        "title": f"{source_title} 被 {target_title} 替代",
        "subject": f"{source_id} 与 {target_id} 的条文替代关系",
        "change_summary": f"依据废止关系数据，{source_id} 当前有 {len(articles)} 条条文被标记为需要核对 {target_id}；新增或修改废止关系后，本主题会自动更新。",
        "applicability_note": "本主题由废止关系表自动生成，只说明哪些旧条文存在替代风险；同一主题的具体数值仍应查看对应现行条文。",
        "keywords": [source_id, target_id, source_title, target_title, "替代", "废止"],
        "versions": [
            _registry_version(
                registry,
                source_id,
                status="legacy_design",
                status_label="旧版/被替代",
                requirement=f"{source_title} 中 {len(articles)} 条条文被标记为需核对 {target_title}；示例条文：{'、'.join(samples)}。",
                source_excerpt=f"废止关系：{source_id} -> {target_id}，共 {len(articles)} 条，来源 {source_info.get('title') if source_info else '废止关系表'}。",
            ),
            _registry_version(
                registry,
                target_id,
                status="current_general",
                status_label="现行替代规范",
                requirement=f"{target_title} 为当前优先核验对象，相关旧条文不能直接作为最终依据。",
                source_excerpt=f"替代关系：{source_id} 的相关条文应优先核对 {target_id}。",
            ),
        ],
        "replaced_article_count": len(articles),
        "sample_articles": samples,
        "topic_kind": "generated",
        "source": "abolished_articles_seed.json",
    }


def _all_topics() -> list[dict[str, Any]]:
    curated = []
    for topic in load_standard_timeline().get("topics") or []:
        item = dict(topic)
        item["topic_kind"] = "curated"
        item["source"] = "standard_timeline.json"
        curated.append(item)
    registry = load_standard_registry()
    abolished = load_abolished_seed()
    dynamic = [_effective_timeline_topic(registry)]
    dynamic.extend(_replacement_topic(registry, rule, abolished.get("source")) for rule in abolished.get("rules") or [])
    return dynamic + curated


def list_version_topics() -> dict[str, Any]:
    topics: list[dict[str, Any]] = []
    for topic in _all_topics():
        versions = _sort_versions(topic.get("versions") or [])
        topics.append(
            {
                "topic_id": topic.get("topic_id"),
                "title": topic.get("title"),
                "subject": topic.get("subject"),
                "change_summary": topic.get("change_summary"),
                "topic_kind": topic.get("topic_kind"),
                "source": topic.get("source"),
                "version_count": len(versions),
                "effective_range": {
                    "first": versions[0].get("effective_date") if versions else None,
                    "latest": versions[-1].get("effective_date") if versions else None,
                },
                "latest_standard_ids": sorted({str(item.get("standard_id")) for item in versions if item.get("standard_id")}),
            }
        )
    return {
        "version": "standard_timeline_dynamic",
        "updated_at": max(
            [
                load_standard_timeline().get("updated_at", ""),
                load_standard_registry().get("updated_at", ""),
                load_abolished_seed().get("updated_at", ""),
            ],
            default="",
        ),
        "topic_kinds": {
            "generated": "由规范注册表/废止关系自动生成",
            "curated": "原文标注示例，不构成完整检查清单",
        },
        "topics": topics,
    }


def get_version_topic(topic_id: str) -> dict[str, Any]:
    topic = next((item for item in _all_topics() if item.get("topic_id") == topic_id), None)
    if not topic:
        raise KeyError(f"unknown standard timeline topic: {topic_id}")
    versions = _sort_versions(topic.get("versions") or [])
    first = versions[0] if versions else {}
    latest = versions[-1] if versions else {}
    result = dict(topic)
    result["versions"] = versions
    result["comparison"] = {
        "oldest": f"{first.get('standard_id')} {first.get('article_no', '')}".strip() if first else "",
        "latest": f"{latest.get('standard_id')} {latest.get('article_no', '')}".strip() if latest else "",
        "has_change": bool(result.get("change_summary")),
        "status_changed": bool(first and latest and first.get("status") != latest.get("status")),
    }
    return result


def find_version_topics(question: str, *, limit: int = 3, min_score: int = 1) -> list[dict[str, str]]:
    text = re.sub(r"\s+", "", str(question or ""))
    scored: list[tuple[int, dict[str, str]]] = []
    for topic in _all_topics():
        keywords = _clean_keywords(topic.get("keywords"))
        score = sum(1 for keyword in keywords if keyword in text)
        if score >= min_score:
            scored.append(
                (
                    score,
                    {
                        "topic_id": str(topic.get("topic_id")),
                        "title": str(topic.get("title")),
                        "subject": str(topic.get("subject")),
                        "topic_kind": str(topic.get("topic_kind")),
                    },
                )
            )
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:limit]]


def _compact_text(value: Any, limit: int = 700) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _retrieval_version_topic(question: str, results: list[dict[str, Any]], registry: dict[str, Any]) -> dict[str, Any]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for item in results:
        standard_id = item.get("standard_id")
        if not standard_id:
            continue
        article_no = str(item.get("article_no") or "未识别条文")
        group_key = (standard_id, article_no)
        group = groups.get(group_key)
        if not group:
            config = registry.get("standards", {}).get(standard_id) or {}
            role_code = config.get("role") or ""
            role = config.get("category") or ROLE_LABELS.get(role_code) or role_code or "规范"
            is_legacy = role_code in {"specialized_legacy_code", "legacy_design_code"} or "旧版" in str(role)
            group = {
                "standard_id": standard_id,
                "title": config.get("title") or standard_id,
                "article_no": article_no,
                "page": item.get("page"),
                "effective_date": config.get("effective_date") or "",
                "status": "legacy_design" if is_legacy else "current_general",
                "status_label": "旧版/专项" if is_legacy else "现行规范",
                "role": role,
                "requirement": _compact_text(item.get("text")),
                "source_excerpt": _compact_text(item.get("text"), 900),
                "chunk_ids": [],
            }
            groups[group_key] = group
        if item.get("chunk_id") and item["chunk_id"] not in group["chunk_ids"]:
            group["chunk_ids"].append(item["chunk_id"])

    versions = sorted(
        list(groups.values()),
        key=lambda item: (str(item.get("effective_date", "")), str(item.get("article_no", ""))),
    )
    return {
        "topic_id": "query_retrieval_fallback",
        "title": f"“{question}”的版本候选",
        "subject": question,
        "change_summary": f"未命中已标注主题，已从知识库召回 {len(versions)} 个条文候选。同一主题可能还存在其他版本，应以人工核验后的适用版本为准。",
        "applicability_note": "本结果由版本对比检索兜底生成，不等同完整条文历史快照；需要逐条核对条文号、页码、适用条件和现行有效性。",
        "keywords": [question],
        "versions": versions,
        "topic_kind": "retrieval",
        "source": "hybrid_retrieval",
        "matched_by": "retrieval",
    }


def _apply_applicable_date(topic: dict[str, Any], applicable_date: str) -> dict[str, Any]:
    if not applicable_date:
        return topic
    for version in topic.get("versions") or []:
        effective_date = str(version.get("effective_date") or "")
        version["applicable"] = not effective_date or applicable_date >= effective_date
    topic["applicable_date"] = applicable_date
    topic["applicability_note"] = (
        f"已按项目日期 {applicable_date} 标注各版本是否适用；施行日期晚于该日期的版本不能直接作为适用依据。"
    )
    return topic


def resolve_version_question(question: str, *, api_key: str | None = None, top_k: int = 8, applicable_date: str = "") -> dict[str, Any]:
    clean = str(question or "").strip()
    if not clean:
        raise ValueError("对比主题不能为空")
    matched = find_version_topics(clean, limit=1, min_score=2)
    if matched:
        topic = get_version_topic(matched[0]["topic_id"])
        topic["matched_by"] = "topic"
        return _apply_applicable_date(topic, applicable_date)
    from hybrid_retrieval import hybrid_search

    results = hybrid_search(
        clean,
        api_key=api_key,
        use_api_rerank=False,
        final_top_k=max(top_k * 2, 16),
    )
    return _apply_applicable_date(_retrieval_version_topic(clean, results, load_standard_registry()), applicable_date)


def flatten_topic_text(topic: dict[str, Any]) -> str:
    parts = [topic.get("title"), topic.get("change_summary"), topic.get("applicability_note")]
    for version in topic.get("versions") or []:
        parts.extend(
            [
                version.get("requirement"),
                version.get("source_excerpt"),
                f"{version.get('standard_id')} {version.get('article_no', '')} {version.get('effective_date', '')}",
            ]
        )
    return "\n".join(str(part) for part in parts if part)
