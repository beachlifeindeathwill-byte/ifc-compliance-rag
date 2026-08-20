from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "data" / "policy"


@dataclass(frozen=True)
class RouteDecision:
    primary: list[str]
    secondary: list[str]
    reasons: list[str]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_policy() -> tuple[dict, dict]:
    registry = load_json(POLICY_DIR / "standard_registry.json")
    abolished = load_json(POLICY_DIR / "abolished_articles_seed.json")
    return registry, abolished


def normalize_standard_id(value: str) -> str:
    return value.replace("_", " ").strip()


def infer_route(question: str, registry: dict) -> RouteDecision:
    standards = registry["standards"]
    primary: list[str] = []
    secondary: list[str] = []
    reasons: list[str] = []

    explicit_standard_ids: list[str] = []
    for standard_id, item in standards.items():
        aliases = item.get("aliases", [])
        if any(alias in question for alias in aliases):
            primary.append(standard_id)
            explicit_standard_ids.append(standard_id)
            reasons.append(f"问题中明确出现 {item['title']} 或其别名")

    # An explicitly named standard does not exclude another standard whose
    # declared scope is also present in the question. This keeps cross-code
    # applicability data-driven as the registry grows.
    if explicit_standard_ids:
        for standard_id, item in standards.items():
            if standard_id in explicit_standard_ids:
                continue
            matched_scope = [keyword for keyword in item.get("scope_keywords", []) if keyword in question]
            if matched_scope:
                secondary.append(standard_id)
                reasons.append(
                    f"问题同时命中 {item['title']} 的适用范围：{'、'.join(matched_scope[:3])}"
                )

    residential_terms = ["住宅项目", "住宅建筑", "多层住宅", "高层住宅", "居住建筑", "新建住宅", "既有住宅", "住宅户门", "户门", "套内空间"]
    if not explicit_standard_ids and any(term in question for term in residential_terms):
        primary.append("GB 55038-2025")
        if any(term in question for term in ["防火", "疏散", "安全出口"]):
            secondary.append("GB 55037-2022")
        reasons.append("问题集中在住宅项目或住宅户门要求，优先检索现行住宅项目规范")

    if any(term in question for term in ["建筑防火通用规范", "防火通规", "通用要求", "基本目标"]):
        primary.append("GB 55037-2022")
        reasons.append("问题表达为建筑防火通用规范或通用要求")

    general_building_terms = [
        "防火墙",
        "防火隔墙",
        "疏散出口",
        "安全出口",
        "疏散门",
        "疏散照明",
        "消防设施与器材",
        "建筑中设置的消防设施",
        "现行通用要求",
    ]
    design_terms = [
        "厂房",
        "仓库",
        "民用建筑",
        "公共建筑",
        "商业综合体",
        "商业建筑",
        "商店营业厅",
        "中庭",
        "周围连通空间",
        "上下层相连通",
        "防火分隔",
        "防火卷帘",
        "疏散楼梯",
        "防火分区",
        "消防电梯",
        "柴油发电机房",
    ]
    specific_design_terms = [term for term in design_terms if term not in {"民用建筑", "公共建筑"}]
    has_design_scene = any(term in question for term in specific_design_terms)
    if any(term in question for term in general_building_terms):
        target = secondary if has_design_scene and not explicit_standard_ids else primary
        target.append("GB 55037-2022")
        reasons.append("问题命中建筑防火通用要求；存在明确设计场景时作为现行规范复核来源")

    if "消防救援设施" in question or "消防救援" in question:
        primary.append("GB 55037-2022")
        reasons.append("消防救援设施属于建筑防火通用规范重点章节")

    facility_terms = ["消防设施", "消防给水", "消火栓", "自动喷水", "灭火系统", "火灾自动报警", "灭火器", "防烟排烟"]
    if any(term in question for term in facility_terms) and not any(term in question for term in general_building_terms):
        primary.append("GB 55036-2022")
        reasons.append("问题集中在消防设施系统")

    garage_terms = ["汽车库", "修车库", "停车场", "停车数量", "汽车疏散出口"]
    if any(term in question for term in garage_terms):
        primary.append("GB 50067-2014")
        secondary.append("GB 55037-2022")
        reasons.append("问题集中在车库专项场景，同时需要校核建筑防火通用规范")

    if not primary and any(term in question for term in design_terms):
        primary.append("GB 50016-2014")
        secondary.append("GB 55037-2022")
        reasons.append("问题为建筑设计防火细节，同时需要检查通用规范是否替代旧条文")

    primary = dedupe(primary)
    secondary = [item for item in dedupe(secondary) if item not in primary]
    if not primary:
        reasons.append("未识别出明确适用规范，使用全库检索并按规范优先级辅助排序")

    return RouteDecision(primary=primary, secondary=secondary, reasons=reasons)


def dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def is_abolished(chunk: dict, abolished: dict) -> dict | None:
    standard_id = normalize_standard_id(chunk.get("standard_id", ""))
    article_no = chunk.get("article_no") or ""
    if not article_no:
        return None

    for rule in abolished.get("rules", []):
        if rule["source_standard_id"] != standard_id:
            continue
        for article in rule.get("articles", []):
            if article_no == article or article_no.startswith(article + "."):
                return {
                    "source_standard_id": rule["source_standard_id"],
                    "source_article": article,
                    "replaced_by_standard_id": rule["replaced_by_standard_id"],
                    "replaced_by_title": rule["replaced_by_title"],
                }
    return None


def policy_bonus(question: str, chunk: dict, route: RouteDecision, registry: dict, abolished: dict) -> tuple[float, list[str]]:
    standard_id = normalize_standard_id(chunk.get("standard_id", ""))
    standard = registry["standards"].get(standard_id, {})
    notes: list[str] = []
    bonus = 0.0
    explicit_standard_ids = [
        candidate_id
        for candidate_id, item in registry["standards"].items()
        if any(alias in question for alias in item.get("aliases", []))
    ]

    if standard_id in route.primary:
        bonus += float(standard.get("route_bonus", 10.0))
        notes.append("命中规范路由主标准")
    elif standard_id in route.secondary:
        bonus += 4.0
        notes.append("命中规范路由辅助校核标准")

    if any(alias in question for alias in standard.get("aliases", [])):
        bonus += 32.0
        notes.append("问题明确提到该标准")
    elif explicit_standard_ids and standard_id not in route.secondary:
        bonus -= 18.0
        notes.append("问题明确提到其他标准，当前标准降权")

    if any(keyword in question for keyword in standard.get("scope_keywords", [])):
        bonus += 2.5
        notes.append("问题命中该标准适用范围关键词")

    if any(term in question for term in ["通用规范", "通用要求", "基本目标"]):
        bonus += standard.get("priority", 0) * 0.04
        notes.append("通用类问题按规范优先级辅助排序")

    abolished_info = is_abolished(chunk, abolished)
    if abolished_info:
        bonus -= 1.0
        notes.append(f"候选条文在 seed 废止表中，需校核 {abolished_info['replaced_by_standard_id']}")

    return bonus, notes
