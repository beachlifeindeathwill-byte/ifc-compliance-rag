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
from evaluate_retrieval import is_hit, load_jsonl, snippet
from hybrid_retrieval import hybrid_search
from evidence_confidence import confidence_from_results
from field_extraction import field_recall_status
from retrieval_policy import infer_route, is_abolished, load_policy


DEFAULT_EVAL = ROOT / "data" / "eval_sets" / "fire_code_focus_single_turn.jsonl"
OUT_DIR = ROOT / "data" / "retrieval_runs"


def expected_hit(case: dict, chunk: dict) -> bool:
    accepted = case.get("accepted_targets") or []
    if accepted:
        return any(
            is_hit(
                {
                    "expected_standard_id": target.get("standard_id"),
                    "expected_article": target.get("article"),
                    "expected_keywords": target.get("keywords", case.get("expected_keywords", [])),
                },
                chunk,
            )
            for target in accepted
        )
    if not case.get("expected_standard_id") and not case.get("expected_article"):
        return False
    probe = {
        "expected_standard_id": case.get("expected_standard_id"),
        "expected_article": case.get("expected_article"),
        "expected_keywords": case.get("expected_keywords", []),
    }
    return is_hit(probe, chunk)


def standard_hit(case: dict, chunk: dict) -> bool:
    accepted = case.get("accepted_targets") or []
    if accepted:
        return any(chunk.get("standard_id") == target.get("standard_id") for target in accepted)
    expected = case.get("expected_standard_id")
    return bool(expected) and chunk.get("standard_id") == expected


def article_hit(case: dict, chunk: dict) -> bool:
    accepted = case.get("accepted_targets") or []
    if accepted:
        article = chunk.get("article_no") or ""
        return any(
            target.get("article")
            and (article.startswith(target["article"]) or (target["article"].endswith(".0") and article.startswith(target["article"][:-2] + ".")))
            for target in accepted
        )
    expected = case.get("expected_article")
    if not expected:
        return False
    article = chunk.get("article_no") or ""
    if article.startswith(expected):
        return True
    return expected.endswith(".0") and article.startswith(expected[:-2] + ".")


def first_rank(results: list[dict], predicate) -> int | None:
    for rank, chunk in enumerate(results, start=1):
        if predicate(chunk):
            return rank
    return None


def supporting_target_rank(results: list[dict], target: dict) -> int | None:
    probe = {
        "expected_standard_id": target.get("standard_id"),
        "expected_article": target.get("article"),
        "expected_keywords": target.get("keywords", []),
    }
    return first_rank(results, lambda chunk: is_hit(probe, chunk))


def at_k(rank: int | None, k: int) -> bool:
    return bool(rank and rank <= k)


def is_scored_case(case: dict) -> bool:
    if case.get("score_retrieval") is False:
        return False
    if case.get("type") == "boundary":
        return False
    return bool(case.get("accepted_targets") or case.get("expected_standard_id") or case.get("expected_article"))


def has_numeric_requirement(text: str) -> bool:
    return bool(re.search(r"\d+(?:\.\d+)?\s*(?:m|h|㎡|m2|人|层|%)", text or "", re.IGNORECASE))


def asks_numeric_question(question: str) -> bool:
    return bool(re.search(r"多少|几|不应大于|不应小于|最大|最小|净宽|面积|距离|高度|耐火|倍", question or ""))


def failure_analysis(case: dict, results: list[dict], confidence: dict, first_hit_rank: int | None, first_standard_rank: int | None, first_article_rank: int | None, field_status: list[dict], abolished: dict) -> dict:
    categories: list[str] = []
    notes: list[str] = []

    expected_behavior = case.get("expected_behavior")
    if expected_behavior:
        categories.append(f"boundary:{expected_behavior}")

    if is_scored_case(case):
        if first_hit_rank is None:
            categories.append("expected_not_in_top5")
            notes.append("期望标准/条文没有进入 Top-5，属于召回失败或候选扩展不足。")
        elif first_hit_rank > 1:
            categories.append("expected_not_top1")
            notes.append(f"正确依据在第 {first_hit_rank} 位，说明排序或重排仍需优化。")
        if first_standard_rank is None:
            categories.append("standard_miss")
            notes.append("期望标准没有进入 Top-5，优先检查规范识别和 metadata 加权。")
        elif first_standard_rank > 1:
            categories.append("standard_not_top1")
            notes.append(f"期望标准在第 {first_standard_rank} 位，存在跨规范竞争。")
        if case.get("expected_article") and first_article_rank is None:
            categories.append("article_miss")
            notes.append("期望条文没有进入 Top-5，优先检查切片、条文号继承和表格索引。")
        elif first_article_rank and first_article_rank > 1:
            categories.append("article_not_top1")
            notes.append(f"期望条文在第 {first_article_rank} 位，说明 Top-1 排序不稳。")

    if confidence.get("level") == "low":
        categories.append("low_confidence")
        notes.append(confidence.get("reason", "证据置信度低。"))
    elif confidence.get("level") == "medium":
        categories.append("medium_confidence")

    missing_fields = [item["field"] for item in field_status if not item["ok"]]
    if missing_fields:
        categories.append("missing_fields")
        notes.append("首条证据字段缺失：" + "、".join(missing_fields))

    if results:
        top = results[0]
        if is_abolished(top, abolished):
            categories.append("deprecated_top_candidate")
            notes.append("Top-1 命中旧规范废止/替代风险，需要现行规范校核。")
        if len(results) > 1:
            top_score = float(top.get("hybrid_score") or 0)
            second_score = float(results[1].get("hybrid_score") or 0)
            if top_score - second_score < 3:
                categories.append("small_score_margin")
                notes.append("Top-1 与 Top-2 分差小，候选排序不够稳定。")
        standards = Counter(item.get("standard_id") or "未知标准" for item in results[:5])
        if len(standards) >= 3:
            categories.append("cross_standard_competition")
            notes.append("Top-5 分布在多份规范中，容易被相近关键词吸引到其他文档。")
        if asks_numeric_question(case.get("question", "")) and not any(has_numeric_requirement(item.get("text", "")) for item in results[:3]):
            categories.append("numeric_not_in_top3")
            notes.append("问题问数值，但前三条没有稳定数值字段。")
    else:
        categories.append("no_candidates")
        notes.append("没有召回候选片段。")

    return {
        "categories": list(dict.fromkeys(categories)),
        "notes": list(dict.fromkeys(notes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--env-file", type=Path)
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
    hit_at_1 = hit_at_3 = hit_at_5 = 0
    standard_at_1 = standard_at_3 = standard_at_5 = 0
    article_at_1 = article_at_3 = article_at_5 = 0
    precision_at_1 = precision_at_3 = precision_at_5 = 0
    mrr_total = 0.0
    answerable_high_or_medium = 0
    boundary_low = 0
    attribution_counts: Counter[str] = Counter()
    field_ok_counts: dict[str, int] = {}
    field_total_counts: dict[str, int] = {}
    composite_evidence_cases = 0
    composite_evidence_covered = 0

    for idx, case in enumerate(cases, start=1):
        query = case["question"]
        route = infer_route(query, registry)
        results = hybrid_search(query, api_key=api_key, final_top_k=8)
        confidence = confidence_from_results(results, route.primary + route.secondary, abolished, query)
        top = results[0] if results else {}
        first_hit_rank = first_rank(results, lambda chunk: expected_hit(case, chunk))
        first_standard_rank = first_rank(results, lambda chunk: standard_hit(case, chunk))
        first_article_rank = first_rank(results, lambda chunk: article_hit(case, chunk))
        scored = is_scored_case(case)
        supporting_targets = case.get("supporting_targets") or []
        supporting_ranks = [supporting_target_rank(results[:5], target) for target in supporting_targets]
        if supporting_targets:
            composite_evidence_cases += 1
            if all(supporting_ranks):
                composite_evidence_covered += 1

        if scored and at_k(first_hit_rank, 1):
            hit_at_1 += 1
        if scored and at_k(first_hit_rank, 3):
            hit_at_3 += 1
        if scored and at_k(first_hit_rank, 5):
            hit_at_5 += 1
        if scored and at_k(first_standard_rank, 1):
            standard_at_1 += 1
        if scored and at_k(first_standard_rank, 3):
            standard_at_3 += 1
        if scored and at_k(first_standard_rank, 5):
            standard_at_5 += 1
        if scored and at_k(first_article_rank, 1):
            article_at_1 += 1
        if scored and at_k(first_article_rank, 3):
            article_at_3 += 1
        if scored and at_k(first_article_rank, 5):
            article_at_5 += 1
        top5 = results[:5]
        if scored:
            precision_at_1 += sum(1 for item in top5[:1] if expected_hit(case, item)) / max(1, 1)
            precision_at_3 += sum(1 for item in top5[:3] if expected_hit(case, item)) / max(1, 3)
            precision_at_5 += sum(1 for item in top5[:5] if expected_hit(case, item)) / max(1, 5)
            if first_hit_rank:
                mrr_total += 1 / first_hit_rank
        if confidence["level"] in {"high", "medium"}:
            answerable_high_or_medium += 1
        if case.get("type") == "boundary" and confidence["level"] == "low":
            boundary_low += 1

        field_status = [
            {"field": name, "ok": ok, "value": value}
            for name, ok, value in field_recall_status(top)
        ]
        for field in field_status:
            field_total_counts[field["field"]] = field_total_counts.get(field["field"], 0) + 1
            if field["ok"]:
                field_ok_counts[field["field"]] = field_ok_counts.get(field["field"], 0) + 1
        analysis = failure_analysis(
            case,
            results,
            confidence,
            first_hit_rank,
            first_standard_rank,
            first_article_rank,
            field_status,
            abolished,
        )
        if supporting_targets and not all(supporting_ranks):
            analysis["categories"].append("composite_evidence_missing")
            missing_support = [
                f"{target.get('standard_id')} {target.get('article')}"
                for target, rank in zip(supporting_targets, supporting_ranks)
                if not rank
            ]
            analysis["notes"].append("Top-5 缺少复合结论所需辅助依据：" + "、".join(missing_support))
        attribution_counts.update(analysis["categories"])

        rows.append(
            {
                "id": case["id"],
                "type": case.get("type", "single_turn"),
                "question": query,
                "focus": case.get("focus"),
                "category": case.get("category"),
                "expected_standard_id": case.get("expected_standard_id"),
                "expected_article": case.get("expected_article"),
                "accepted_targets": case.get("accepted_targets", []),
                "expected_behavior": case.get("expected_behavior"),
                "supporting_targets": supporting_targets,
                "supporting_target_ranks": supporting_ranks,
                "first_hit_rank": first_hit_rank,
                "first_standard_rank": first_standard_rank,
                "first_article_rank": first_article_rank,
                "evidence_confidence": confidence,
                "field_recall": field_status,
                "failure_analysis": analysis,
                "top_results": [
                    {
                        "rank": rank,
                        "standard_id": item.get("standard_id"),
                        "article_no": item.get("article_no"),
                        "page": item.get("page"),
                        "hybrid_score": round(float(item.get("hybrid_score") or 0), 4),
                        "is_expected_hit": expected_hit(case, item),
                        "snippet": snippet(item.get("text", ""), limit=240),
                    }
                    for rank, item in enumerate(results[:5], start=1)
                ],
            }
        )
        print(f"evaluated {idx}/{len(cases)} {case['id']} confidence={confidence['level']} hit_rank={first_hit_rank}")

    total = len(cases)
    scored_cases = [case for case in cases if is_scored_case(case)]
    scored_total = max(1, len(scored_cases))
    summary = {
        "cases": total,
        "scored_cases": len(scored_cases),
        "hit_at_1": round(hit_at_1 / scored_total, 3),
        "hit_at_3": round(hit_at_3 / scored_total, 3),
        "hit_at_5": round(hit_at_5 / scored_total, 3),
        "standard_at_1": round(standard_at_1 / scored_total, 3),
        "standard_at_3": round(standard_at_3 / scored_total, 3),
        "standard_at_5": round(standard_at_5 / scored_total, 3),
        "article_at_1": round(article_at_1 / scored_total, 3),
        "article_at_3": round(article_at_3 / scored_total, 3),
        "article_at_5": round(article_at_5 / scored_total, 3),
        "precision_at_1": round(precision_at_1 / scored_total, 3),
        "precision_at_3": round(precision_at_3 / scored_total, 3),
        "precision_at_5": round(precision_at_5 / scored_total, 3),
        "mrr_at_10": round(mrr_total / scored_total, 3),
        "field_recall_rate": {
            field: round(field_ok_counts.get(field, 0) / max(1, total_count), 3)
            for field, total_count in field_total_counts.items()
        },
        "composite_evidence_cases": composite_evidence_cases,
        "composite_evidence_coverage_at_5": round(
            composite_evidence_covered / max(1, composite_evidence_cases), 3
        ),
        "answerable_high_or_medium_rate": round(answerable_high_or_medium / total, 3),
        "boundary_low_expected_count": boundary_low,
        "failure_attribution_counts": dict(attribution_counts.most_common()),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "focus_single_turn_eval.json"
    out_md = OUT_DIR / "focus_single_turn_eval.md"
    out_json.write_text(json.dumps({"summary": summary, "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 聚焦单轮检索评估",
        "",
        f"- 用例数：{summary['cases']}",
        f"- 可计分用例数：{summary['scored_cases']}",
        f"- Hit@1: {summary['hit_at_1']}",
        f"- Hit@3: {summary['hit_at_3']}",
        f"- Hit@5: {summary['hit_at_5']}",
        f"- Standard@1/@3/@5: {summary['standard_at_1']} / {summary['standard_at_3']} / {summary['standard_at_5']}",
        f"- Article@1/@3/@5: {summary['article_at_1']} / {summary['article_at_3']} / {summary['article_at_5']}",
        f"- Precision@1/@3/@5: {summary['precision_at_1']} / {summary['precision_at_3']} / {summary['precision_at_5']}",
        f"- MRR@10: {summary['mrr_at_10']}",
        f"- 高/中置信可回答比例：{summary['answerable_high_or_medium_rate']}",
        f"- 字段召回率：{json.dumps(summary['field_recall_rate'], ensure_ascii=False)}",
        f"- 复合证据覆盖率@5：{summary['composite_evidence_coverage_at_5']} "
        f"（{summary['composite_evidence_cases']} 个用例）",
        "",
        "## 失败归因",
        "",
        *[f"- {name}: {count}" for name, count in attribution_counts.most_common()],
        "",
        "## 用例",
        "",
    ]
    for row in rows:
        lines.append(f"### {row['id']} {row['question']}")
        lines.append(f"- Focus: {row['focus']}")
        lines.append(f"- Expected: {row.get('expected_standard_id')} / {row.get('expected_article')} / {row.get('expected_behavior')}")
        lines.append(f"- First hit rank: {row['first_hit_rank']}")
        lines.append(f"- First standard/article rank: {row['first_standard_rank']} / {row['first_article_rank']}")
        if row["supporting_targets"]:
            lines.append(f"- Supporting evidence ranks: {row['supporting_target_ranks']}")
        lines.append(f"- Evidence confidence: {row['evidence_confidence']['level']} - {row['evidence_confidence']['reason']}")
        lines.append(f"- Failure categories: {', '.join(row['failure_analysis']['categories']) or 'none'}")
        for note in row["failure_analysis"]["notes"][:4]:
            lines.append(f"  - {note}")
        fields = ", ".join([f"{x['field']}={x['value']}" for x in row["field_recall"]])
        lines.append(f"- Field recall: {fields}")
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
