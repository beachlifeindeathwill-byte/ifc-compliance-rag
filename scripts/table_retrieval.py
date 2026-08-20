from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from evaluate_retrieval import load_jsonl, tokenize


ROOT = Path(__file__).resolve().parents[1]
TABLE_INDEX_PATH = ROOT / "data" / "structured_tables" / "table_index.jsonl"

TABLE_REF_RE = re.compile(r"(?:表|续表)\s*([0-9]+(?:\.[0-9]+)+(?:-\d+)?)")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:m2|m²|㎡|m'|m|h|%|倍|层|人)?", re.I)

TABLE_QUERY_TERMS = [
    "表",
    "最大",
    "最小",
    "允许",
    "面积",
    "净宽",
    "宽度",
    "耐火等级",
    "耐火极限",
    "防火分区",
    "地下",
    "半地下",
    "高层",
    "单多层",
    "单、多层",
    "汽车库",
    "设备用房",
    "疏散楼梯",
    "疏散走道",
    "疏散门",
]

STOP_TOKENS = set("的是了和与及在中内外有无为按应不小大多少什么哪些这个那个如果")
RELATION_QUERY_TERMS = (
    "增加",
    "减少",
    "放宽",
    "折减",
    "自动灭火",
    "自动喷水",
    "局部设置",
    "倍",
)


@lru_cache(maxsize=1)
def load_table_index() -> list[dict]:
    if not TABLE_INDEX_PATH.exists():
        return []
    return load_jsonl(TABLE_INDEX_PATH)


def query_table_refs(query: str) -> list[str]:
    return TABLE_REF_RE.findall(query)


def looks_like_table_query(query: str) -> bool:
    if query_table_refs(query):
        return True
    strong_terms = ["最大", "最小", "允许", "面积", "耐火等级", "耐火极限", "每 100人", "每100人"]
    if not any(term in query for term in strong_terms):
        return False
    hits = sum(1 for term in TABLE_QUERY_TERMS if term in query)
    return hits >= 2


def content_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOP_TOKENS and len(token.strip()) > 0}


def score_table(query: str, table: dict, route_primary: list[str] | None = None) -> tuple[float, list[str]]:
    notes: list[str] = []
    score = 0.0
    route_primary = route_primary or []
    table_no = table.get("table_no")
    refs = query_table_refs(query)

    if refs:
        if table_no in refs:
            score += 80.0
            notes.append(f"table: 命中显式表号 {table_no}")
        else:
            score -= 15.0

    if route_primary and table.get("standard_id") in route_primary:
        score += 8.0
        notes.append("table: 命中规范路由")

    query_terms = [term for term in TABLE_QUERY_TERMS if term in query]
    text = f"{table.get('title', '')} {table.get('context_text', '')}"

    category_terms = ["厂房", "仓库", "冷库", "汽车库", "修车库", "民用建筑", "公共建筑", "住宅建筑", "设备用房"]
    query_categories = [term for term in category_terms if term in query]
    table_categories = [term for term in category_terms if term in text]
    if query_categories:
        matched_categories = [term for term in query_categories if term in text]
        if matched_categories:
            score += min(12.0, len(matched_categories) * 6.0)
            notes.append(f"table: 命中建筑类别 {'/'.join(matched_categories[:3])}")
        elif table_categories:
            score -= 10.0
            notes.append("table: 建筑类别不匹配")
    else:
        unrelated_special_categories = [term for term in ["厂房", "仓库", "冷库", "汽车库", "修车库"] if term in text]
        if unrelated_special_categories and "建筑" in query:
            score -= 8.0
            notes.append(f"table: 未指定专项类别，降低 {'/'.join(unrelated_special_categories[:3])} 表")

    matched_terms = [term for term in query_terms if term in text]
    if matched_terms:
        score += min(24.0, len(matched_terms) * 4.0)
        notes.append(f"table: 命中字段 {'/'.join(matched_terms[:5])}")

    q_tokens = content_tokens(query)
    t_tokens = content_tokens(text)
    overlap = q_tokens & t_tokens
    if overlap:
        score += min(18.0, len(overlap) * 0.6)

    q_numbers = set(NUMBER_RE.findall(query))
    if q_numbers:
        table_numbers = set(table.get("numbers") or [])
        number_overlap = q_numbers & table_numbers
        if number_overlap:
            score += 8.0
            notes.append(f"table: 命中数值 {'/'.join(sorted(number_overlap))}")

    if table.get("has_extracted_grid"):
        score += 4.0
        notes.append("table: 有抽取表格块")

    if table.get("is_continuation"):
        score += 1.0
        notes.append("table: 续表片段")

    return score, notes


def search_tables(query: str, *, route_primary: list[str] | None = None, top_k: int = 8) -> list[dict]:
    if not looks_like_table_query(query):
        return []

    scored = []
    for table in load_table_index():
        score, notes = score_table(query, table, route_primary=route_primary)
        if score <= 0:
            continue
        item = dict(table)
        item["table_score"] = score
        item["table_notes"] = notes
        scored.append(item)

    scored.sort(key=lambda item: item["table_score"], reverse=True)
    return scored[:top_k]


def table_to_candidate(table: dict) -> dict:
    text = table.get("context_text") or table.get("text") or ""
    candidate = {
        "chunk_id": table["table_id"],
        "standard_id": table["standard_id"],
        "standard_title": table.get("standard_title"),
        "source_file": table.get("source_file"),
        "page": table.get("page"),
        "article_no": table.get("article_no") or table.get("table_no"),
        "has_table": True,
        "chunk_type": "structured_table",
        "table_no": table.get("table_no"),
        "text": text,
        "row_id": f"table:{table['table_id']}",
        "fused_score": 0.0,
        "bm25_score": 0.0,
        "vector_score": None,
        "table_score": table.get("table_score", 0.0),
        "hybrid_score": table.get("table_score", 0.0),
        "policy_notes": table.get("table_notes", []),
    }
    return candidate


def related_evidence_candidates(query: str, tables: list[dict], *, top_k: int = 6) -> list[dict]:
    """Return independently citable clauses related to retrieved table bases."""
    query_tokens = content_tokens(query)
    relation_terms = [term for term in RELATION_QUERY_TERMS if term in query]
    if not relation_terms:
        return []

    candidates: dict[str, dict] = {}
    for table in tables:
        for related in table.get("related_evidence") or []:
            chunk_id = related.get("chunk_id")
            if not chunk_id:
                continue
            text = related.get("text", "") or ""
            matched_relations = [term for term in relation_terms if term in text]
            if not matched_relations:
                continue
            overlap = query_tokens & content_tokens(text)
            # Keep the table's base value as the primary evidence while making
            # a separately citable modifier competitive for the remaining slots.
            score = 84.0 + min(24.0, len(matched_relations) * 8.0) + min(18.0, len(overlap) * 0.8)
            score += min(12.0, float(table.get("table_score") or 0.0) * 0.1)
            item = {
                "chunk_id": chunk_id,
                "standard_id": table.get("standard_id"),
                "standard_title": table.get("standard_title"),
                "source_file": table.get("source_file"),
                "page": related.get("page"),
                "article_no": related.get("article_no"),
                "has_table": False,
                "chunk_type": "table_related_clause",
                "table_no": table.get("table_no"),
                "text": text,
                "row_id": f"table-related:{chunk_id}",
                "fused_score": 0.0,
                "bm25_score": 0.0,
                "vector_score": None,
                "table_score": table.get("table_score", 0.0),
                "hybrid_score": score,
                "policy_notes": [
                    f"table: 关联表 {table.get('table_no')} 的独立调整条文",
                    f"table: 命中调整条件 {'/'.join(matched_relations[:3])}",
                ],
            }
            previous = candidates.get(chunk_id)
            if previous is None or item["hybrid_score"] > previous["hybrid_score"]:
                candidates[chunk_id] = item
    return sorted(candidates.values(), key=lambda item: item["hybrid_score"], reverse=True)[:top_k]
