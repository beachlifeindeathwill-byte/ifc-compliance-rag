from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

from evaluate_retrieval import load_jsonl
from query_intent import QueryIntent, detect_query_intent
from retrieval_policy import RouteDecision, normalize_standard_id


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "processed_text" / "chunks.jsonl"


@lru_cache(maxsize=1)
def load_chunks() -> list[dict]:
    return load_jsonl(CHUNKS_PATH)


def _route_bonus(chunk: dict, route: RouteDecision) -> float:
    standard_id = normalize_standard_id(chunk.get("standard_id", ""))
    if standard_id in route.primary:
        return 15.0
    if standard_id in route.secondary:
        return 6.0
    return 0.0


def _article_match(article: str | None, ref: str) -> bool:
    article = article or ""
    return article == ref or article.startswith(ref + ".") or ref.startswith(article + ".")


def search_direct_locators(
    question: str,
    *,
    route: RouteDecision,
    top_k: int = 12,
    intent: QueryIntent | None = None,
) -> list[dict]:
    intent = intent or detect_query_intent(question)
    if not intent.asks_locator:
        return []

    candidates: list[dict] = []
    for chunk in load_chunks():
        score = 0.0
        notes: list[str] = []

        for ref in intent.article_refs:
            if _article_match(chunk.get("article_no"), ref):
                score += 80.0
                notes.append(f"direct: 命中条文号 {ref}")

        page = chunk.get("page")
        if page is not None:
            for ref in intent.page_refs:
                if int(page) == ref:
                    score += 55.0
                    notes.append(f"direct: 命中页码 {ref}")

        text = chunk.get("text", "") or ""
        table_no = chunk.get("table_no")
        for ref in intent.table_refs:
            table_near_head = bool(re.search(rf"^\s*(?:续)?表\s*{re.escape(ref)}", text[:160]))
            structured_table = "[TABLE_START]" in text and bool(re.search(rf"(?:表|续表)\s*{re.escape(ref)}", text[:220]))
            if table_no == ref or table_near_head or structured_table:
                score += 70.0
                notes.append(f"direct: 命中表号 {ref}")

        if score <= 0:
            continue

        score += _route_bonus(chunk, route)
        if chunk.get("article_no"):
            score += 4.0
        if chunk.get("has_table"):
            score += 2.0

        item = dict(chunk)
        item["hybrid_score"] = score
        item["direct_score"] = score
        item["policy_notes"] = notes
        candidates.append(item)

    candidates.sort(key=lambda item: item["hybrid_score"], reverse=True)
    return candidates[:top_k]
