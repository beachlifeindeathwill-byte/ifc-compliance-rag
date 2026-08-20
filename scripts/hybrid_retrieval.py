from __future__ import annotations

import json
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np

from build_faiss_vectorstore import local_hash_embedding, post_embeddings
from evaluate_retrieval import bm25_score, build_index, load_jsonl, tokenize
from retrieval_policy import infer_route, is_abolished, load_policy, policy_bonus
from evaluate_policy_retrieval import detect_building_types, local_rerank_bonus
from direct_retrieval import search_direct_locators
from query_expansion import expand_query_for_retrieval
from query_intent import detect_query_intent
from qdrant_retriever import qdrant_search_if_enabled
from siliconflow_rerank import SiliconFlowRerankError, rerank_documents
from table_retrieval import related_evidence_candidates, search_tables, table_to_candidate
from text_normalization import normalize_for_retrieval


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "processed_text" / "chunks.jsonl"
VECTORSTORE_DIR = ROOT / "data" / "vectorstore"


CONTINUATION_RE = re.compile(r"^\s*(?:[1-9]\d{0,1}\s+|[（(]?[一二三四五六七八九十1-9]\s*[）)]|续表|注[:：])")


def looks_like_article_continuation(text: str) -> bool:
    head = (text or "").strip()[:120]
    return bool(CONTINUATION_RE.search(head))


def looks_like_orphaned_article_body(text: str, previous_text: str | None) -> bool:
    if not previous_text:
        return False
    previous = previous_text.strip()
    current = (text or "").strip()
    if not current or re.search(r"^\s*\d+(?:\.\d+){1,3}\s+", current):
        return False
    if len(previous) <= 80 and not re.search(r"[。；;：:]$", previous):
        return True
    body_starters = ("安全出口", "疏散", "防火", "消防", "建筑", "房间", "场所", "道路", "门", "窗")
    return len(previous) <= 120 and current.startswith(body_starters)


def annotate_article_context(chunks: list[dict]) -> list[dict]:
    annotated: list[dict] = []
    last_article_by_standard: dict[str, tuple[str, int, str]] = {}

    for row_id, raw in enumerate(chunks):
        chunk = dict(raw)
        chunk["text"] = normalize_for_retrieval(chunk.get("text", ""))
        standard_id = chunk.get("standard_id") or ""
        article_no = chunk.get("article_no")

        if article_no:
            if (
                chunk.get("inherited_article_no")
                and standard_id in last_article_by_standard
                and last_article_by_standard[standard_id][0] == article_no
            ):
                inherited_article, parent_row_id, _ = last_article_by_standard[standard_id]
                chunk["parent_article_no"] = inherited_article
                chunk["parent_row_id"] = parent_row_id
            else:
                chunk["parent_article_no"] = article_no
                chunk["parent_row_id"] = row_id
                last_article_by_standard[standard_id] = (article_no, row_id, chunk.get("text", ""))
        elif standard_id in last_article_by_standard and (
            looks_like_article_continuation(chunk.get("text", ""))
            or looks_like_orphaned_article_body(chunk.get("text", ""), last_article_by_standard[standard_id][2])
        ):
            inherited_article, parent_row_id, _ = last_article_by_standard[standard_id]
            chunk["article_no"] = inherited_article
            chunk["parent_article_no"] = inherited_article
            chunk["parent_row_id"] = parent_row_id
            chunk["inherited_article_no"] = True

        annotated.append(chunk)

    return annotated


def enrich_parent_context(candidate: dict, chunks: list[dict], *, max_chars: int = 2600) -> dict:
    row_id = candidate.get("row_id")
    parent_row_id = candidate.get("parent_row_id")
    if row_id is None or parent_row_id is None or row_id == parent_row_id:
        return candidate
    if not (0 <= int(row_id) < len(chunks) and 0 <= int(parent_row_id) < len(chunks)):
        return candidate

    parent = chunks[int(parent_row_id)]
    current = chunks[int(row_id)]
    parts = []
    for item in (parent, current):
        text = (item.get("text") or "").strip()
        if text and text not in parts:
            parts.append(text)
    merged = "\n\n".join(parts).strip()
    if len(merged) <= len(candidate.get("text", "")) + 30:
        return candidate

    updated = dict(candidate)
    updated["child_text"] = candidate.get("text")
    updated["text"] = merged[:max_chars]
    updated["policy_notes"] = list(updated.get("policy_notes") or []) + ["parent-context: 合并同条文上下文"]
    return updated


def enrich_article_neighbors(candidate: dict, chunks: list[dict], *, max_chars: int = 2600) -> dict:
    """Join adjacent chunks that belong to the same article across page breaks."""
    row_id = candidate.get("row_id")
    article_no = candidate.get("article_no")
    standard_id = candidate.get("standard_id")
    try:
        row_index = int(row_id)
    except (TypeError, ValueError):
        return candidate
    if not article_no or not (0 <= row_index < len(chunks)):
        return candidate

    start = max(0, row_index - 2)
    end = min(len(chunks), row_index + 3)
    neighbors = [
        item
        for item in chunks[start:end]
        if item.get("standard_id") == standard_id and item.get("article_no") == article_no
    ]
    if len(neighbors) < 2:
        return candidate

    parts: list[str] = []
    for item in neighbors:
        text = (item.get("text") or "").strip()
        if text and text not in parts:
            parts.append(text)
    merged = "\n\n".join(parts).strip()
    if len(merged) <= len(candidate.get("text", "")) + 30:
        return candidate

    updated = dict(candidate)
    updated["child_text"] = candidate.get("text")
    updated["text"] = merged[:max_chars]
    updated["policy_notes"] = list(updated.get("policy_notes") or []) + ["article-context: 合并同条文相邻分页"]
    return updated


def inject_replacement_cross_checks(candidates: list[dict], abolished: dict) -> list[dict]:
    if not candidates:
        return candidates

    injected: list[dict] = []
    used_chunk_ids: set[str] = set()
    replacement_targets: list[str] = []

    for item in candidates[:3]:
        abolished_info = is_abolished(item, abolished)
        if abolished_info:
            target = abolished_info["replaced_by_standard_id"]
            if target not in replacement_targets:
                replacement_targets.append(target)

    for item in candidates:
        chunk_id = item.get("chunk_id")
        if chunk_id not in used_chunk_ids:
            injected.append(item)
            used_chunk_ids.add(chunk_id)

        abolished_info = is_abolished(item, abolished)
        if not abolished_info:
            continue

        target = abolished_info["replaced_by_standard_id"]
        replacement = next(
            (
                candidate
                for candidate in candidates
                if candidate.get("standard_id") == target and candidate.get("chunk_id") not in used_chunk_ids
            ),
            None,
        )
        if replacement:
            replacement = dict(replacement)
            replacement["policy_notes"] = list(replacement.get("policy_notes") or []) + [
                f"replacement-check: 校核 {abolished_info['source_standard_id']} {abolished_info['source_article']}"
            ]
            injected.append(replacement)
            used_chunk_ids.add(replacement.get("chunk_id"))

    for target in replacement_targets:
        if any(item.get("standard_id") == target for item in injected[:5]):
            continue
        replacement = next(
            (
                candidate
                for candidate in candidates
                if candidate.get("standard_id") == target and candidate.get("chunk_id") not in used_chunk_ids
            ),
            None,
        )
        if replacement:
            replacement = dict(replacement)
            replacement["policy_notes"] = list(replacement.get("policy_notes") or []) + ["replacement-check: 补充现行通用规范候选"]
            injected.insert(min(1, len(injected)), replacement)
            used_chunk_ids.add(replacement.get("chunk_id"))

    return injected


def inject_routed_standard_evidence(
    candidates: list[dict],
    route,
    registry: dict,
    *,
    coverage_k: int = 5,
) -> list[dict]:
    """Keep one independently citable candidate for each routed standard."""
    if not candidates or coverage_k < 2:
        return candidates

    routed = []
    for standard_id in list(route.primary) + list(route.secondary):
        if standard_id in registry.get("standards", {}) and standard_id not in routed:
            routed.append(standard_id)

    result = list(candidates)
    for standard_id in routed:
        if any(item.get("standard_id") == standard_id for item in result[:coverage_k]):
            continue
        companion = next((item for item in candidates if item.get("standard_id") == standard_id), None)
        if not companion:
            continue
        companion = dict(companion)
        companion["policy_notes"] = list(companion.get("policy_notes") or []) + [
            "route-coverage: 补充已命中适用范围的规范证据"
        ]
        companion_id = companion.get("chunk_id")
        result = [item for item in result if item.get("chunk_id") != companion_id]
        result.insert(min(coverage_k - 1, len(result)), companion)
    return result


@lru_cache(maxsize=1)
def load_vectorstore() -> tuple[faiss.Index, list[dict], dict]:
    index_path = VECTORSTORE_DIR / "faiss.index"
    meta_path = VECTORSTORE_DIR / "chunk_meta.jsonl"
    config_path = VECTORSTORE_DIR / "embedding_config.json"
    if not index_path.exists() or not meta_path.exists() or not config_path.exists():
        raise FileNotFoundError("FAISS vectorstore is missing. Run scripts/build_faiss_vectorstore.py first.")
    fd, temp_index_path = tempfile.mkstemp(prefix="bim_fire_faiss_read_", suffix=".index")
    os.close(fd)
    temp_path = Path(temp_index_path)
    try:
        temp_path.write_bytes(index_path.read_bytes())
        index = faiss.read_index(str(temp_path))
    finally:
        temp_path.unlink(missing_ok=True)
    metas = load_jsonl(meta_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return index, metas, config


@lru_cache(maxsize=1)
def load_bm25_resources() -> tuple[list[dict], list[list[str]], dict[str, int], float]:
    chunks = annotate_article_context(load_jsonl(CHUNKS_PATH))
    tokenized_docs, df, avgdl = build_index(chunks)
    return chunks, tokenized_docs, df, avgdl


def embed_query(query: str, *, api_key: str | None, config: dict) -> np.ndarray:
    if config["embedding_provider"] == "local-hash":
        vector = local_hash_embedding(query, dimension=int(config["dimension"]))
    else:
        if not api_key:
            raise RuntimeError("SILICONFLOW_API_KEY is required for this vectorstore")
        vector = post_embeddings([query], api_key=api_key, model=config["embedding_model"], url=config["embedding_url"], timeout=60, retries=3)[0]
    array = np.asarray([vector], dtype="float32")
    faiss.normalize_L2(array)
    return array


def rrf_fuse(rankings: list[tuple[list[int], float]], *, k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking, weight in rankings:
        for rank, row_id in enumerate(ranking, start=1):
            scores[row_id] = scores.get(row_id, 0.0) + weight / (k + rank)
    return scores


def hybrid_search(
    query: str,
    *,
    api_key: str | None = None,
    use_api_rerank: bool = False,
    rerank_api_key: str | None = None,
    rerank_top_n: int = 8,
    bm25_top_k: int = 80,
    vector_top_k: int = 40,
    fused_top_k: int = 80,
    final_top_k: int = 8,
) -> list[dict]:
    chunks, tokenized_docs, df, avgdl = load_bm25_resources()
    retrieval_query = expand_query_for_retrieval(query)
    query_tokens = tokenize(retrieval_query)

    bm25_scored = []
    for row_id, (chunk, doc_tokens) in enumerate(zip(chunks, tokenized_docs)):
        score = bm25_score(query_tokens, doc_tokens, df, len(chunks), avgdl)
        if chunk.get("article_no"):
            score += 0.2
        if score > 0:
            bm25_scored.append((score, row_id))
    bm25_scored.sort(reverse=True)
    bm25_ranking = [row_id for _, row_id in bm25_scored[:bm25_top_k]]
    bm25_scores = {row_id: score for score, row_id in bm25_scored}

    index, metas, config = load_vectorstore()
    query_vector = embed_query(retrieval_query, api_key=api_key, config=config)
    vector_ranking = qdrant_search_if_enabled(query_vector, top_k=vector_top_k)
    if vector_ranking is None:
        distances, indices = index.search(query_vector, vector_top_k)
        vector_ranking = [int(row_id) for row_id in indices[0] if row_id >= 0]
        vector_scores = {int(row_id): float(score) for row_id, score in zip(indices[0], distances[0]) if row_id >= 0}
    else:
        vector_scores = {}

    fused = rrf_fuse([(bm25_ranking, 2.0), (vector_ranking, 0.6)])
    registry, abolished = load_policy()
    route = infer_route(query, registry)
    intent = detect_query_intent(query)

    candidates = []
    for row_id, fused_score in sorted(fused.items(), key=lambda item: item[1], reverse=True)[:fused_top_k]:
        chunk = chunks[row_id]
        scoring_chunk = dict(chunk)
        scoring_chunk["row_id"] = row_id
        scoring_chunk = enrich_parent_context(scoring_chunk, chunks)
        policy_score, policy_notes = policy_bonus(query, scoring_chunk, route, registry, abolished)
        local_score, local_notes = local_rerank_bonus(query, scoring_chunk)
        bm25_component = bm25_scores.get(row_id, 0.0)
        vector_component = (vector_scores.get(row_id) or 0.0) * 2.0
        final_score = bm25_component + policy_score + local_score + vector_component + fused_score * 10.0
        candidate = dict(chunk)
        candidate["row_id"] = row_id
        candidate["fused_score"] = fused_score
        candidate["bm25_score"] = bm25_component
        candidate["vector_score"] = vector_scores.get(row_id)
        candidate["policy_notes"] = policy_notes + local_notes
        candidate["building_types"] = detect_building_types(chunk.get("text", ""))
        candidate["hybrid_score"] = final_score
        candidate["retrieval_query"] = retrieval_query
        candidates.append(candidate)

    table_hits = search_tables(retrieval_query, route_primary=route.primary, top_k=12)
    existing_by_chunk_id = {item.get("chunk_id"): item for item in candidates}
    explicit_table_query = bool(re.search(r"(?:表|续表)\s*[0-9]+(?:\.[0-9]+)+(?:-\d+)?", query))
    for table in table_hits:
        table_candidate = table_to_candidate(table)
        existing = existing_by_chunk_id.get(table.get("chunk_id"))
        if existing:
            boost = min(20.0, float(table.get("table_score") or 0.0) * 0.35)
            existing["hybrid_score"] += boost
            existing["table_score"] = table.get("table_score")
            existing["policy_notes"] = list(existing.get("policy_notes") or []) + list(table.get("table_notes") or [])
        else:
            table_score = float(table.get("table_score") or 0.0)
            table_candidate["hybrid_score"] = table_score if explicit_table_query else table_score * 0.7
            table_candidate["retrieval_query"] = retrieval_query
            candidates.append(table_candidate)

    existing_by_chunk_id = {item.get("chunk_id"): item for item in candidates}
    for related_candidate in related_evidence_candidates(retrieval_query, table_hits, top_k=8):
        existing = existing_by_chunk_id.get(related_candidate.get("chunk_id"))
        if existing:
            existing["hybrid_score"] += min(10.0, float(related_candidate.get("hybrid_score") or 0.0) * 0.12)
            existing["policy_notes"] = list(existing.get("policy_notes") or []) + list(
                related_candidate.get("policy_notes") or []
            )
        else:
            related_candidate["retrieval_query"] = retrieval_query
            candidates.append(related_candidate)
            existing_by_chunk_id[related_candidate.get("chunk_id")] = related_candidate

    existing_by_chunk_id = {item.get("chunk_id"): item for item in candidates}
    for direct_candidate in search_direct_locators(query, route=route, intent=intent, top_k=12):
        existing = existing_by_chunk_id.get(direct_candidate.get("chunk_id"))
        if existing:
            boost = min(35.0, float(direct_candidate.get("direct_score") or 0.0) * 0.4)
            existing["hybrid_score"] += boost
            existing["direct_score"] = direct_candidate.get("direct_score")
            existing["policy_notes"] = list(existing.get("policy_notes") or []) + list(direct_candidate.get("policy_notes") or [])
        else:
            direct_candidate["retrieval_query"] = retrieval_query
            candidates.append(direct_candidate)
            existing_by_chunk_id[direct_candidate.get("chunk_id")] = direct_candidate

    candidates = [enrich_parent_context(item, chunks) for item in candidates]
    candidates = [enrich_article_neighbors(item, chunks) for item in candidates]
    candidates.sort(key=lambda item: item["hybrid_score"], reverse=True)
    candidates = inject_replacement_cross_checks(candidates, abolished)
    candidates = inject_routed_standard_evidence(candidates, route, registry)
    candidates = [enrich_article_neighbors(item, chunks) for item in candidates]
    if use_api_rerank:
        api_candidates = candidates[: min(30, len(candidates))]
        try:
            api_results = rerank_documents(
                query,
                [item["text"] for item in api_candidates],
                api_key=rerank_api_key or api_key,
                top_n=min(max(1, rerank_top_n), len(api_candidates)),
            )
        except SiliconFlowRerankError as exc:
            if candidates:
                notes = candidates[0].setdefault("policy_notes", [])
                notes.append(f"api rerank fallback: {exc}")
            return candidates[:final_top_k]

        reranked_candidates = []
        for result in api_results:
            item = dict(api_candidates[result.index])
            item["api_rerank_score"] = result.relevance_score
            item["policy_notes"] = list(item.get("policy_notes") or []) + ["api rerank: SiliconFlow"]
            reranked_candidates.append(item)
        if reranked_candidates:
            reranked_with_coverage = inject_routed_standard_evidence(
                reranked_candidates + api_candidates,
                route,
                registry,
            )
            return reranked_with_coverage[:final_top_k]

    return candidates[:final_top_k]
