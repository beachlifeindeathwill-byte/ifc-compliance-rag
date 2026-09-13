from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from build_faiss_vectorstore import load_env_file
from conversation_context import (
    ConversationState,
    is_meta_reference_question,
    merge_evidence_results,
    normalize_reference_context,
    prefers_context_evidence,
    rank_context_evidence,
    recent_evidence,
    remember_evidence,
    remember_question,
    update_state_from_results,
)
from context_router import ContextDecision, route_context_with_llm
from evaluate_retrieval import is_hit, load_jsonl, snippet
from field_extraction import field_recall_status
from hybrid_retrieval import hybrid_search
from retrieval_app import confidence_from_results
from retrieval_policy import infer_route, load_policy


DEFAULT_EVAL = ROOT / "data" / "eval_sets" / "fire_code_natural_scenarios.jsonl"
OUT_DIR = ROOT / "data" / "retrieval_runs"


BOUNDARY_BEHAVIORS = {"insufficient_context", "knowledge_boundary"}


def expected_hit(case: dict, chunk: dict) -> bool:
    if not case.get("expected_standard_id") and not case.get("expected_article"):
        return False
    probe = {
        "expected_standard_id": case.get("expected_standard_id"),
        "expected_article": case.get("expected_article"),
        "expected_keywords": case.get("expected_keywords", []),
    }
    return is_hit(probe, chunk)


def standard_hit(case: dict, chunk: dict) -> bool:
    expected = case.get("expected_standard_id")
    return bool(expected) and chunk.get("standard_id") == expected


def article_hit(case: dict, chunk: dict) -> bool:
    expected = case.get("expected_article")
    if not expected:
        return False
    article = chunk.get("article_no") or ""
    return article.startswith(expected) or (expected.endswith(".0") and article.startswith(expected[:-2] + "."))


def first_rank(results: list[dict], predicate) -> int | None:
    for rank, chunk in enumerate(results, start=1):
        if predicate(chunk):
            return rank
    return None


def at_k(rank: int | None, k: int) -> bool:
    return bool(rank and rank <= k)


def keyword_coverage(case: dict, results: list[dict], *, top_k: int = 5) -> dict:
    keywords = [str(item) for item in case.get("expected_keywords", []) if str(item).strip()]
    text = "\n".join([item.get("text", "") or "" for item in results[:top_k]])
    hits = [keyword for keyword in keywords if keyword in text]
    return {
        "expected": keywords,
        "hits": hits,
        "rate": round(len(hits) / max(1, len(keywords)), 3),
    }


def is_scored_single_turn(case: dict) -> bool:
    if case.get("type") != "single_turn":
        return False
    if case.get("expected_behavior") in BOUNDARY_BEHAVIORS:
        return False
    return bool(case.get("expected_standard_id") or case.get("expected_article"))


def boundary_pass(case: dict, confidence: dict) -> bool | None:
    behavior = case.get("expected_behavior")
    if behavior not in BOUNDARY_BEHAVIORS:
        return None
    return confidence.get("level") == "low" or not confidence.get("can_answer")


def extract_focus_expectations(case: dict) -> tuple[set[str], set[str]]:
    focus_text = " ".join([str(item) for item in case.get("expected_focus", [])])
    standards = set(re.findall(r"GB\s*\d{5}-\d{4}", focus_text))
    articles = set(re.findall(r"\b\d+(?:\.\d+){1,3}\b", focus_text))
    return standards, articles


def multi_turn_focus_hit(case: dict, turn_rows: list[dict]) -> dict:
    expected_standards, expected_articles = extract_focus_expectations(case)
    if not expected_standards and not expected_articles:
        return {"evaluated": False, "hit": None, "expected_standards": [], "expected_articles": []}

    top_results = []
    for row in turn_rows:
        top_results.extend(row["top_results"])
    standards = {item.get("standard_id") for item in top_results}
    articles = {item.get("article_no") for item in top_results if item.get("article_no")}
    standard_ok = not expected_standards or bool(expected_standards & standards)
    article_ok = not expected_articles or any(
        any(str(article).startswith(expected) for article in articles)
        for expected in expected_articles
    )
    return {
        "evaluated": True,
        "hit": bool(standard_ok and article_ok),
        "expected_standards": sorted(expected_standards),
        "expected_articles": sorted(expected_articles),
    }


def evaluate_single_turn(case: dict, *, api_key: str | None, top_k: int, registry: dict, abolished: dict) -> dict:
    query = case["question"]
    route = infer_route(query, registry)
    results = hybrid_search(query, api_key=api_key, final_top_k=top_k)
    confidence = confidence_from_results(results, route.primary + route.secondary, abolished, query)
    top = results[0] if results else {}
    field_status = [
        {"field": name, "ok": ok, "value": value}
        for name, ok, value in field_recall_status(top)
    ]
    coverage = keyword_coverage(case, results)
    return {
        "id": case["id"],
        "scenario": case.get("scenario"),
        "type": case.get("type"),
        "question": query,
        "expected_standard_id": case.get("expected_standard_id"),
        "expected_article": case.get("expected_article"),
        "expected_behavior": case.get("expected_behavior"),
        "first_hit_rank": first_rank(results, lambda item: expected_hit(case, item)),
        "first_standard_rank": first_rank(results, lambda item: standard_hit(case, item)),
        "first_article_rank": first_rank(results, lambda item: article_hit(case, item)),
        "boundary_pass": boundary_pass(case, confidence),
        "keyword_coverage": coverage,
        "evidence_confidence": confidence,
        "field_recall": field_status,
        "top_results": [
            {
                "rank": rank,
                "standard_id": item.get("standard_id"),
                "article_no": item.get("article_no"),
                "page": item.get("page"),
                "source_file": item.get("source_file"),
                "hybrid_score": round(float(item.get("hybrid_score") or 0), 4),
                "is_expected_hit": expected_hit(case, item),
                "snippet": snippet(item.get("text", ""), limit=240),
            }
            for rank, item in enumerate(results[:5], start=1)
        ],
    }


def route_turn_context(question: str, state: ConversationState) -> ContextDecision:
    try:
        decision = route_context_with_llm(question, state)
        return normalize_reference_context(question, decision, state)
    except Exception as exc:
        return normalize_reference_context(
            question,
            ContextDecision(
                mode="new_topic",
                use_previous_context=False,
                resolved_question=question,
                reason=f"Context Router 不可用，评估按独立问题保守处理：{str(exc)[:120]}",
                source="fallback",
            ),
            state,
        )


def evaluate_multi_turn(case: dict, *, api_key: str | None, top_k: int, registry: dict, abolished: dict) -> dict:
    state = ConversationState()
    turn_rows = []
    rewrite_used = 0
    low_confidence_turns = 0
    answerable_turns = 0

    for round_no, question in enumerate(case["turns"], start=1):
        context_decision = route_turn_context(question, state)
        contextual_followup = bool(context_decision.use_previous_context)
        rewritten = context_decision.resolved_question if contextual_followup and context_decision.resolved_question else question
        rewrite_reasons = [context_decision.reason or ("Context Router 判定为追问" if contextual_followup else "Context Router 判定为新话题")]
        remember_question(state, question, rewritten)
        route = infer_route(rewritten, registry)
        fresh_results = hybrid_search(rewritten, api_key=api_key, final_top_k=top_k)
        context_results = []
        if contextual_followup:
            context_results = recent_evidence(state, sets=2 if is_meta_reference_question(question) else 1, max_items=top_k)
            context_results = rank_context_evidence(question, context_results)
        if context_results and prefers_context_evidence(question):
            results = merge_evidence_results(context_results, fresh_results, max_items=max(top_k, 10))[:top_k]
        elif context_results:
            results = merge_evidence_results(fresh_results, context_results, max_items=max(top_k, 10))[:top_k]
        else:
            results = fresh_results
        update_state_from_results(state, results)
        confidence = confidence_from_results(results, route.primary + route.secondary, abolished, rewritten)
        if not is_meta_reference_question(question):
            remember_evidence(state, results, max_items=top_k)
        if rewritten != question or rewrite_reasons:
            rewrite_used += 1
        if confidence.get("level") == "low":
            low_confidence_turns += 1
        if confidence.get("can_answer"):
            answerable_turns += 1

        turn_rows.append(
            {
                "round": round_no,
                "question": question,
                "rewritten_question": rewritten,
                "rewrite_reasons": rewrite_reasons,
                "context_decision": {
                    "mode": context_decision.mode,
                    "use_previous_context": context_decision.use_previous_context,
                    "reason": context_decision.reason,
                    "source": context_decision.source,
                },
                "contextual_followup": contextual_followup,
                "context_evidence_count": len(context_results),
                "confidence": confidence,
                "top_results": [
                    {
                        "rank": rank,
                        "standard_id": item.get("standard_id"),
                        "article_no": item.get("article_no"),
                        "page": item.get("page"),
                        "source_file": item.get("source_file"),
                        "hybrid_score": round(float(item.get("hybrid_score") or 0), 4),
                        "snippet": snippet(item.get("text", ""), limit=220),
                    }
                    for rank, item in enumerate(results[:5], start=1)
                ],
            }
        )

    focus_hit = multi_turn_focus_hit(case, turn_rows)
    return {
        "id": case["id"],
        "scenario": case.get("scenario"),
        "type": case.get("type"),
        "expected_focus": case.get("expected_focus", []),
        "focus_hit": focus_hit,
        "rewrite_used_turns": rewrite_used,
        "low_confidence_turns": low_confidence_turns,
        "answerable_turns": answerable_turns,
        "turns": turn_rows,
    }


def build_summary(cases: list[dict], rows: list[dict]) -> dict:
    single_rows = [row for row in rows if row.get("type") == "single_turn"]
    multi_rows = [row for row in rows if row.get("type") == "multi_turn"]
    scored_ids = {case["id"] for case in cases if is_scored_single_turn(case)}
    scored_rows = [row for row in single_rows if row["id"] in scored_ids]
    boundary_rows = [row for row in single_rows if row.get("boundary_pass") is not None]
    confidence_counts = Counter(row["evidence_confidence"]["level"] for row in single_rows)

    field_ok_counts: Counter[str] = Counter()
    field_total_counts: Counter[str] = Counter()
    for row in single_rows:
        for field in row.get("field_recall", []):
            field_total_counts[field["field"]] += 1
            if field["ok"]:
                field_ok_counts[field["field"]] += 1

    focus_eval_rows = [row for row in multi_rows if row["focus_hit"]["evaluated"]]
    total_multi_turns = sum(len(row["turns"]) for row in multi_rows)
    context_modes = Counter()
    context_sources = Counter()
    context_followups = 0
    for row in multi_rows:
        for turn in row.get("turns", []):
            decision = turn.get("context_decision") or {}
            context_modes[decision.get("mode") or "unknown"] += 1
            context_sources[decision.get("source") or "unknown"] += 1
            if decision.get("use_previous_context"):
                context_followups += 1
    return {
        "cases": len(cases),
        "single_turn_cases": len(single_rows),
        "multi_turn_cases": len(multi_rows),
        "multi_turn_turns": total_multi_turns,
        "scored_single_turn_cases": len(scored_rows),
        "hit_at_1": round(sum(at_k(row["first_hit_rank"], 1) for row in scored_rows) / max(1, len(scored_rows)), 3),
        "hit_at_3": round(sum(at_k(row["first_hit_rank"], 3) for row in scored_rows) / max(1, len(scored_rows)), 3),
        "hit_at_5": round(sum(at_k(row["first_hit_rank"], 5) for row in scored_rows) / max(1, len(scored_rows)), 3),
        "standard_at_1": round(sum(at_k(row["first_standard_rank"], 1) for row in scored_rows) / max(1, len(scored_rows)), 3),
        "article_at_1": round(sum(at_k(row["first_article_rank"], 1) for row in scored_rows) / max(1, len(scored_rows)), 3),
        "boundary_cases": len(boundary_rows),
        "boundary_pass_rate": round(sum(bool(row["boundary_pass"]) for row in boundary_rows) / max(1, len(boundary_rows)), 3),
        "single_turn_confidence_distribution": dict(confidence_counts),
        "field_recall_rate": {
            field: round(field_ok_counts[field] / max(1, total), 3)
            for field, total in field_total_counts.items()
        },
        "multi_turn_focus_cases": len(focus_eval_rows),
        "multi_turn_focus_hit_rate": round(sum(bool(row["focus_hit"]["hit"]) for row in focus_eval_rows) / max(1, len(focus_eval_rows)), 3),
        "multi_turn_answerable_rate": round(sum(row["answerable_turns"] for row in multi_rows) / max(1, total_multi_turns), 3),
        "multi_turn_low_confidence_turns": sum(row["low_confidence_turns"] for row in multi_rows),
        "context_router_modes": dict(context_modes),
        "context_router_sources": dict(context_sources),
        "context_router_followup_turns": context_followups,
    }


def write_markdown(summary: dict, rows: list[dict], path: Path) -> None:
    lines = [
        "# Natural Scenario RAG Evaluation",
        "",
        "本报告只评估检索和证据置信度，不把标准答案写入线上检索逻辑。",
        "",
        "## Summary",
        "",
        f"- Cases: {summary['cases']}",
        f"- Single-turn cases: {summary['single_turn_cases']}",
        f"- Multi-turn scenarios / turns: {summary['multi_turn_cases']} / {summary['multi_turn_turns']}",
        f"- Scored single-turn cases: {summary['scored_single_turn_cases']}",
        f"- Hit@1 / Hit@3 / Hit@5: {summary['hit_at_1']} / {summary['hit_at_3']} / {summary['hit_at_5']}",
        f"- Standard@1 / Article@1: {summary['standard_at_1']} / {summary['article_at_1']}",
        f"- Boundary pass rate: {summary['boundary_pass_rate']} ({summary['boundary_cases']} cases)",
        f"- Multi-turn focus hit rate: {summary['multi_turn_focus_hit_rate']} ({summary['multi_turn_focus_cases']} scenarios)",
        f"- Multi-turn answerable rate: {summary['multi_turn_answerable_rate']}",
        f"- Context router modes: {json.dumps(summary.get('context_router_modes', {}), ensure_ascii=False)}",
        f"- Context router sources: {json.dumps(summary.get('context_router_sources', {}), ensure_ascii=False)}",
        f"- Field recall: {json.dumps(summary['field_recall_rate'], ensure_ascii=False)}",
        f"- Confidence distribution: {json.dumps(summary['single_turn_confidence_distribution'], ensure_ascii=False)}",
        "",
        "## Single-Turn Cases",
        "",
    ]
    for row in [item for item in rows if item.get("type") == "single_turn"]:
        lines.append(f"### {row['id']} {row['question']}")
        lines.append(f"- Expected: {row.get('expected_standard_id')} / {row.get('expected_article')} / {row.get('expected_behavior')}")
        lines.append(f"- First hit / standard / article rank: {row['first_hit_rank']} / {row['first_standard_rank']} / {row['first_article_rank']}")
        lines.append(f"- Boundary pass: {row['boundary_pass']}")
        lines.append(f"- Keyword coverage: {row['keyword_coverage']['rate']} ({'、'.join(row['keyword_coverage']['hits'])})")
        lines.append(f"- Confidence: {row['evidence_confidence']['level']} - {row['evidence_confidence']['reason']}")
        for item in row["top_results"]:
            lines.append(
                f"  - #{item['rank']} {item['standard_id']} article={item['article_no']} "
                f"p{item['page']} expected_hit={item['is_expected_hit']} score={item['hybrid_score']}"
            )
            lines.append(f"    {item['snippet']}")
        lines.append("")

    lines.extend(["## Multi-Turn Cases", ""])
    for row in [item for item in rows if item.get("type") == "multi_turn"]:
        focus = row["focus_hit"]
        lines.append(f"### {row['id']}")
        lines.append(f"- Expected focus: {'；'.join(row.get('expected_focus') or [])}")
        lines.append(f"- Focus hit: {focus['hit']} (standards={focus['expected_standards']}, articles={focus['expected_articles']})")
        lines.append(f"- Rewrite used turns: {row['rewrite_used_turns']}")
        lines.append(f"- Answerable / low-confidence turns: {row['answerable_turns']} / {row['low_confidence_turns']}")
        for turn in row["turns"]:
            lines.append(f"  - Round {turn['round']}: {turn['question']}")
            if turn["rewritten_question"] != turn["question"]:
                lines.append(f"    Rewritten: {turn['rewritten_question']}")
            if turn.get("context_decision"):
                decision = turn["context_decision"]
                lines.append(
                    f"    Context router: mode={decision.get('mode')} use_previous={decision.get('use_previous_context')} "
                    f"source={decision.get('source')} reason={decision.get('reason')}"
                )
            lines.append(f"    Confidence: {turn['confidence']['level']} - {turn['confidence']['reason']}")
            if turn["top_results"]:
                top = turn["top_results"][0]
                lines.append(f"    Top1: {top['standard_id']} / {top['article_no']} / p{top['page']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.env_file:
        load_env_file(args.env_file)
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("SILICONFLOW_API_KEY is not set")

    registry, abolished = load_policy()
    cases = load_jsonl(args.eval_file)
    if args.limit:
        cases = cases[: args.limit]

    rows = []
    for idx, case in enumerate(cases, start=1):
        if case.get("type") == "multi_turn":
            row = evaluate_multi_turn(case, api_key=api_key, top_k=args.top_k, registry=registry, abolished=abolished)
            print(f"evaluated {idx}/{len(cases)} {case['id']} multi_turn focus_hit={row['focus_hit']['hit']}")
        else:
            row = evaluate_single_turn(case, api_key=api_key, top_k=args.top_k, registry=registry, abolished=abolished)
            print(f"evaluated {idx}/{len(cases)} {case['id']} confidence={row['evidence_confidence']['level']} hit_rank={row['first_hit_rank']}")
        rows.append(row)

    summary = build_summary(cases, rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "natural_scenarios_eval.json"
    out_md = OUT_DIR / "natural_scenarios_eval.md"
    out_json.write_text(json.dumps({"summary": summary, "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, rows, out_md)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
