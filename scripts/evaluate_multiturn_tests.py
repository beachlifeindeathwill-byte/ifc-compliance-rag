from __future__ import annotations

import argparse
import json
import os
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
from evaluate_retrieval import is_hit, snippet
from hybrid_retrieval import hybrid_search
from retrieval_app import confidence_from_results
from retrieval_policy import infer_route, load_policy


DEFAULT_EVAL = ROOT / "data" / "eval_sets" / "fire_code_multiturn_cases.json"
OUT_DIR = ROOT / "data" / "retrieval_runs"


def load_cases(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_hit(turn: dict, chunk: dict) -> bool:
    if turn.get("expected_standard_id") and not turn.get("expected_article") and not turn.get("expected_keywords"):
        return chunk.get("standard_id") == turn.get("expected_standard_id")
    probe = {
        "expected_standard_id": turn.get("expected_standard_id"),
        "expected_article": turn.get("expected_article"),
        "expected_keywords": turn.get("expected_keywords", []),
    }
    if probe["expected_standard_id"] or probe["expected_article"]:
        return is_hit(probe, chunk)
    return False


def behavior_hit(expected_behavior: str | None, results: list[dict], rewritten_question: str) -> tuple[bool | None, str]:
    if not expected_behavior:
        return None, "无行为型断言"
    text = "\n".join([item.get("text", "") or "" for item in results[:5]])
    standards = {item.get("standard_id") for item in results[:5]}
    articles = {item.get("article_no") for item in results[:5]}

    if expected_behavior == "must_show_base_value_and_multiplier":
        ok = "GB 50067-2014" in standards and ("2.0倍" in text or "2倍" in text) and ("5.1.2" in articles or "5.1.1" in articles)
        return ok, "检查是否召回车库专项规范、自动灭火系统倍数和相关条文"
    if expected_behavior == "must_show_formula":
        ok = "4000" in rewritten_question and "地下汽车库" in rewritten_question and ("5.1.1" in articles or "5.1.2" in articles)
        return ok, "检查是否保留面积计算场景并召回面积依据"
    if expected_behavior == "must_back_reference_article":
        ok = bool({"5.1.1", "5.1.2"} & articles) or "GB 50067-2014" in standards
        return ok, "检查是否能回溯上一轮车库面积依据"
    if expected_behavior == "must_retrieve_related_fire_separation_article":
        ok = bool({"5.3.2", "6.2.4"} & articles) or "自动扶梯" in text
        return ok, "检查是否召回自动扶梯/上下连通开口相关条款"
    if expected_behavior == "must_state_if_evidence_missing_or_needs_more_conditions":
        ok = bool(results) and ("疏散楼梯" in rewritten_question or "疏散楼梯" in text)
        return ok, "检查是否围绕当前场景召回疏散楼梯证据，最终仍需回答层说明条件不足"
    return False, f"未知行为断言：{expected_behavior}"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    if args.env_file:
        load_env_file(args.env_file)
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("SILICONFLOW_API_KEY is not set")

    registry, abolished = load_policy()
    spec = load_cases(args.eval_file)
    cases = spec["cases"]
    if args.limit_cases:
        cases = cases[: args.limit_cases]

    case_rows = []
    retrieval_assertions = 0
    retrieval_hits = 0
    behavior_assertions = 0
    behavior_hits = 0
    low_confidence_turns = 0
    context_modes: Counter[str] = Counter()
    context_sources: Counter[str] = Counter()
    context_followup_turns = 0

    for case_idx, case in enumerate(cases, start=1):
        state = ConversationState()
        turn_rows = []
        print(f"case {case_idx}/{len(cases)} {case['id']} {case['name']}")
        for turn in case["turns"]:
            question = turn["question"]
            context_decision = route_turn_context(question, state)
            contextual_followup = bool(context_decision.use_previous_context)
            rewritten = context_decision.resolved_question if contextual_followup and context_decision.resolved_question else question
            rewrite_reasons = [context_decision.reason or ("Context Router 判定为追问" if contextual_followup else "Context Router 判定为新话题")]
            remember_question(state, question, rewritten)
            route = infer_route(rewritten, registry)
            fresh_results = hybrid_search(rewritten, api_key=api_key, final_top_k=args.top_k)
            context_results = []
            if contextual_followup:
                context_results = recent_evidence(state, sets=2 if is_meta_reference_question(question) else 1, max_items=args.top_k)
                context_results = rank_context_evidence(question, context_results)
            if context_results and prefers_context_evidence(question):
                results = merge_evidence_results(context_results, fresh_results, max_items=max(args.top_k, 10))[: args.top_k]
            elif context_results:
                results = merge_evidence_results(fresh_results, context_results, max_items=max(args.top_k, 10))[: args.top_k]
            else:
                results = fresh_results
            update_state_from_results(state, results)
            confidence = confidence_from_results(results, route.primary + route.secondary, abolished, rewritten)
            if not is_meta_reference_question(question):
                remember_evidence(state, results, max_items=args.top_k)
            if confidence["level"] == "low":
                low_confidence_turns += 1
            context_modes[context_decision.mode] += 1
            context_sources[context_decision.source] += 1
            if context_decision.use_previous_context:
                context_followup_turns += 1

            first_hit_rank = None
            for rank, item in enumerate(results, start=1):
                if expected_hit(turn, item):
                    first_hit_rank = rank
                    break
            if turn.get("expected_standard_id") or turn.get("expected_article"):
                retrieval_assertions += 1
                if first_hit_rank and first_hit_rank <= 5:
                    retrieval_hits += 1

            behavior_ok, behavior_reason = behavior_hit(turn.get("expected_behavior"), results, rewritten)
            if behavior_ok is not None:
                behavior_assertions += 1
                if behavior_ok:
                    behavior_hits += 1

            turn_rows.append(
                {
                    "round": turn["round"],
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
                    "expected_article": turn.get("expected_article"),
                    "expected_standard_id": turn.get("expected_standard_id"),
                    "expected_behavior": turn.get("expected_behavior"),
                    "first_hit_rank": first_hit_rank,
                    "behavior_hit": behavior_ok,
                    "behavior_reason": behavior_reason,
                    "confidence": confidence,
                    "top_results": [
                        {
                            "rank": rank,
                            "standard_id": item.get("standard_id"),
                            "article_no": item.get("article_no"),
                            "page": item.get("page"),
                            "hybrid_score": round(float(item.get("hybrid_score") or 0), 4),
                            "is_expected_hit": expected_hit(turn, item),
                            "snippet": snippet(item.get("text", ""), limit=220),
                        }
                        for rank, item in enumerate(results[:5], start=1)
                    ],
                }
            )
            print(
                f"  round {turn['round']} confidence={confidence['level']} "
                f"hit_rank={first_hit_rank} behavior={behavior_ok}"
            )

        case_rows.append(
            {
                "id": case["id"],
                "name": case["name"],
                "goal": case.get("goal"),
                "turns": turn_rows,
            }
        )

    total_turns = sum(len(case["turns"]) for case in cases)
    summary = {
        "version": spec.get("version"),
        "cases": len(cases),
        "turns": total_turns,
        "retrieval_assertions": retrieval_assertions,
        "retrieval_hit_at_5": round(retrieval_hits / max(1, retrieval_assertions), 3),
        "behavior_assertions": behavior_assertions,
        "behavior_hit_rate": round(behavior_hits / max(1, behavior_assertions), 3),
        "low_confidence_turns": low_confidence_turns,
        "context_router_modes": dict(context_modes),
        "context_router_sources": dict(context_sources),
        "context_router_followup_turns": context_followup_turns,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "multiturn_eval.json"
    out_md = OUT_DIR / "multiturn_eval.md"
    out_json.write_text(json.dumps({"summary": summary, "results": case_rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Multi-Turn Retrieval Evaluation",
        "",
        f"- Version: `{summary['version']}`",
        f"- Cases: {summary['cases']}",
        f"- Turns: {summary['turns']}",
        f"- Retrieval assertions: {summary['retrieval_assertions']}",
        f"- Retrieval Hit@5: {summary['retrieval_hit_at_5']}",
        f"- Behavior assertions: {summary['behavior_assertions']}",
        f"- Behavior hit rate: {summary['behavior_hit_rate']}",
        f"- Low confidence turns: {summary['low_confidence_turns']}",
        f"- Context router modes: {json.dumps(summary.get('context_router_modes', {}), ensure_ascii=False)}",
        f"- Context router sources: {json.dumps(summary.get('context_router_sources', {}), ensure_ascii=False)}",
        "",
    ]
    for case in case_rows:
        lines.append(f"## {case['id']} {case['name']}")
        lines.append(case.get("goal") or "")
        lines.append("")
        for row in case["turns"]:
            lines.append(f"### Round {row['round']}")
            lines.append(f"- Original: {row['question']}")
            lines.append(f"- Rewritten: {row['rewritten_question']}")
            lines.append(f"- Rewrite reasons: {'；'.join(row['rewrite_reasons']) or '无'}")
            decision = row.get("context_decision") or {}
            lines.append(
                f"- Context router: mode={decision.get('mode')} use_previous={decision.get('use_previous_context')} "
                f"source={decision.get('source')} reason={decision.get('reason')}"
            )
            lines.append(f"- Expected: {row.get('expected_standard_id')} / {row.get('expected_article')} / {row.get('expected_behavior')}")
            lines.append(f"- First hit rank: {row['first_hit_rank']}")
            lines.append(f"- Behavior hit: {row['behavior_hit']} ({row['behavior_reason']})")
            lines.append(f"- Confidence: {row['confidence']['level']} - {row['confidence']['reason']}")
            for item in row["top_results"]:
                lines.append(
                    f"  - #{item['rank']} {item['standard_id']} article={item['article_no']} "
                    f"p{item['page']} expected_hit={item['is_expected_hit']} score={item['hybrid_score']}"
                )
                lines.append(f"    {item['snippet']}")
            lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
