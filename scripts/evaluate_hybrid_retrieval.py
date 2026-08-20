from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from build_faiss_vectorstore import load_env_file
from evaluate_retrieval import is_hit, load_jsonl, snippet
from hybrid_retrieval import VECTORSTORE_DIR, hybrid_search


ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "data" / "eval_sets" / "fire_code_eval_v2.jsonl"
OUT_DIR = ROOT / "data" / "retrieval_runs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.env_file:
        load_env_file(args.env_file)
    config = json.loads((VECTORSTORE_DIR / "embedding_config.json").read_text(encoding="utf-8"))
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if config["embedding_provider"] == "siliconflow" and not api_key:
        raise RuntimeError("SILICONFLOW_API_KEY is not set")

    questions = load_jsonl(EVAL_PATH)
    if args.limit:
        questions = questions[: args.limit]

    results = []
    hit_at_1 = hit_at_3 = hit_at_5 = 0
    standard_at_1 = standard_at_3 = standard_at_5 = 0
    mrr_total = 0.0

    for idx, question in enumerate(questions, start=1):
        top = hybrid_search(question["question"], api_key=api_key, final_top_k=8)
        first_hit_rank = None
        first_standard_rank = None
        for rank, chunk in enumerate(top, start=1):
            if first_standard_rank is None and chunk["standard_id"] == question["expected_standard_id"]:
                first_standard_rank = rank
            if first_hit_rank is None and is_hit(question, chunk):
                first_hit_rank = rank

        if first_hit_rank == 1:
            hit_at_1 += 1
        if first_hit_rank and first_hit_rank <= 3:
            hit_at_3 += 1
        if first_hit_rank and first_hit_rank <= 5:
            hit_at_5 += 1
        if first_hit_rank:
            mrr_total += 1 / first_hit_rank

        if first_standard_rank == 1:
            standard_at_1 += 1
        if first_standard_rank and first_standard_rank <= 3:
            standard_at_3 += 1
        if first_standard_rank and first_standard_rank <= 5:
            standard_at_5 += 1

        results.append(
            {
                "id": question["id"],
                "question": question["question"],
                "expected_standard_id": question["expected_standard_id"],
                "expected_article": question.get("expected_article"),
                "first_hit_rank": first_hit_rank,
                "first_standard_rank": first_standard_rank,
                "top_results": [
                    {
                        "rank": rank,
                        "standard_id": chunk["standard_id"],
                        "article_no": chunk.get("article_no"),
                        "page": chunk.get("page"),
                        "chunk_id": chunk["chunk_id"],
                        "hybrid_score": round(chunk["hybrid_score"], 4),
                        "bm25_score": round(chunk.get("bm25_score", 0.0), 4),
                        "vector_score": None if chunk.get("vector_score") is None else round(chunk["vector_score"], 4),
                        "fused_score": round(chunk["fused_score"], 4),
                        "is_hit": is_hit(question, chunk),
                        "policy_notes": chunk.get("policy_notes", []),
                        "snippet": snippet(chunk["text"]),
                    }
                    for rank, chunk in enumerate(top[:5], start=1)
                ],
            }
        )
        print(f"evaluated {idx}/{len(questions)} {question['id']}")

    total = len(questions)
    summary = {
        "questions": total,
        "hit_at_1": round(hit_at_1 / total, 3),
        "hit_at_3": round(hit_at_3 / total, 3),
        "hit_at_5": round(hit_at_5 / total, 3),
        "mrr_at_8": round(mrr_total / total, 3),
        "standard_at_1": round(standard_at_1 / total, 3),
        "standard_at_3": round(standard_at_3 / total, 3),
        "standard_at_5": round(standard_at_5 / total, 3),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "retrieval_eval_v3_hybrid.json").write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Retrieval Evaluation v3 Hybrid",
        "",
        f"- Eval set: `{EVAL_PATH.name}`",
        f"- Questions: {summary['questions']}",
        f"- Hit@1: {summary['hit_at_1']}",
        f"- Hit@3: {summary['hit_at_3']}",
        f"- Hit@5: {summary['hit_at_5']}",
        f"- MRR@8: {summary['mrr_at_8']}",
        f"- Standard@1: {summary['standard_at_1']}",
        f"- Standard@3: {summary['standard_at_3']}",
        f"- Standard@5: {summary['standard_at_5']}",
        "",
        "## Weak Cases",
        "",
    ]
    for result in results:
        if result["first_hit_rank"] is None or result["first_hit_rank"] > 3:
            lines.append(f"### {result['id']} {result['question']}")
            lines.append(f"Expected: {result['expected_standard_id']} / {result['expected_article']}")
            lines.append(f"First hit rank: {result['first_hit_rank']}")
            lines.append(f"First standard rank: {result['first_standard_rank']}")
            for item in result["top_results"]:
                lines.append(
                    f"- #{item['rank']} {item['standard_id']} p{item['page']} article={item['article_no']} "
                    f"hit={item['is_hit']} hybrid={item['hybrid_score']} vector={item['vector_score']} `{item['chunk_id']}`"
                )
                lines.append(f"  {item['snippet']}")
            lines.append("")
    (OUT_DIR / "retrieval_eval_v3_hybrid.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
