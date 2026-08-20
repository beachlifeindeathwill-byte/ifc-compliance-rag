from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "processed_text" / "chunks.jsonl"
DEFAULT_EVAL_PATH = ROOT / "data" / "eval_sets" / "fire_code_eval_v2.jsonl"
FALLBACK_EVAL_PATH = ROOT / "data" / "eval_sets" / "fire_code_eval_v1.jsonl"
EVAL_PATH = DEFAULT_EVAL_PATH if DEFAULT_EVAL_PATH.exists() else FALLBACK_EVAL_PATH
OUT_DIR = ROOT / "data" / "retrieval_runs"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*", text.lower())


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_index(chunks: list[dict]) -> tuple[list[list[str]], dict[str, int], float]:
    tokenized_docs = [tokenize(c["text"]) for c in chunks]
    df: Counter[str] = Counter()
    for toks in tokenized_docs:
        df.update(set(toks))
    avgdl = sum(len(toks) for toks in tokenized_docs) / max(1, len(tokenized_docs))
    return tokenized_docs, dict(df), avgdl


def bm25_score(query_tokens: list[str], doc_tokens: list[str], df: dict[str, int], total_docs: int, avgdl: float) -> float:
    k1 = 1.5
    b = 0.75
    counts = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for token in query_tokens:
        if token not in counts:
            continue
        n = df.get(token, 0)
        idf = math.log(1 + (total_docs - n + 0.5) / (n + 0.5))
        tf = counts[token]
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
    return score


def score_chunk(chunk: dict, query_tokens: list[str], doc_tokens: list[str], df: dict[str, int], total_docs: int, avgdl: float) -> float:
    score = bm25_score(query_tokens, doc_tokens, df, total_docs, avgdl)

    if chunk.get("has_table"):
        score += 0.15
    if chunk.get("article_no"):
        score += 0.2
    return score


def is_hit(question: dict, chunk: dict) -> bool:
    if question.get("expected_standard_id") and chunk["standard_id"] != question["expected_standard_id"]:
        return False

    expected_article = question.get("expected_article")
    if expected_article:
        article = chunk.get("article_no") or ""
        if article.startswith(expected_article):
            return True
        # Section-level questions use expected articles like 7.0 or 4.0.
        if expected_article.endswith(".0") and article.startswith(expected_article[:-2] + "."):
            return True
        return False

    keywords = question.get("expected_keywords", [])
    if keywords and all(k in chunk["text"] for k in keywords[:2]):
        return True
    return False


def snippet(text: str, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def main() -> None:
    chunks = load_jsonl(CHUNKS_PATH)
    questions = load_jsonl(EVAL_PATH)
    tokenized_docs, df, avgdl = build_index(chunks)
    total_docs = len(chunks)

    results = []
    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    mrr_total = 0.0

    for question in questions:
        query_tokens = tokenize(question["question"])
        scored = []
        for chunk, doc_tokens in zip(chunks, tokenized_docs):
            score = score_chunk(chunk, query_tokens, doc_tokens, df, total_docs, avgdl)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[:10]

        first_hit_rank = None
        for idx, (_, chunk) in enumerate(top, start=1):
            if is_hit(question, chunk):
                first_hit_rank = idx
                break

        if first_hit_rank == 1:
            hit_at_1 += 1
        if first_hit_rank and first_hit_rank <= 3:
            hit_at_3 += 1
        if first_hit_rank and first_hit_rank <= 5:
            hit_at_5 += 1
        if first_hit_rank:
            mrr_total += 1 / first_hit_rank

        results.append(
            {
                "id": question["id"],
                "question": question["question"],
                "expected_standard_id": question.get("expected_standard_id"),
                "expected_article": question.get("expected_article"),
                "first_hit_rank": first_hit_rank,
                "top_results": [
                    {
                        "rank": rank,
                        "score": round(score, 3),
                        "chunk_id": chunk["chunk_id"],
                        "standard_id": chunk["standard_id"],
                        "page": chunk["page"],
                        "article_no": chunk.get("article_no"),
                        "has_table": chunk.get("has_table"),
                        "is_hit": is_hit(question, chunk),
                        "snippet": snippet(chunk["text"]),
                    }
                    for rank, (score, chunk) in enumerate(top[:5], start=1)
                ],
            }
        )

    total = len(questions)
    summary = {
        "questions": total,
        "hit_at_1": round(hit_at_1 / total, 3),
        "hit_at_3": round(hit_at_3 / total, 3),
        "hit_at_5": round(hit_at_5 / total, 3),
        "mrr_at_10": round(mrr_total / total, 3),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "retrieval_eval_v1.json").write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Retrieval Evaluation v1",
        "",
        f"- Eval set: `{EVAL_PATH.name}`",
        f"- Questions: {summary['questions']}",
        f"- Hit@1: {summary['hit_at_1']}",
        f"- Hit@3: {summary['hit_at_3']}",
        f"- Hit@5: {summary['hit_at_5']}",
        f"- MRR@10: {summary['mrr_at_10']}",
        "",
        "## Failed Or Weak Cases",
        "",
    ]
    for result in results:
        rank = result["first_hit_rank"]
        if rank is None or rank > 3:
            lines.append(f"### {result['id']} {result['question']}")
            lines.append(f"Expected: {result['expected_standard_id']} / {result['expected_article']}")
            lines.append(f"First hit rank: {rank}")
            for item in result["top_results"]:
                lines.append(
                    f"- #{item['rank']} {item['standard_id']} p{item['page']} article={item['article_no']} "
                    f"hit={item['is_hit']} score={item['score']} `{item['chunk_id']}`"
                )
                lines.append(f"  {item['snippet']}")
            lines.append("")

    (OUT_DIR / "retrieval_eval_v1.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {OUT_DIR / 'retrieval_eval_v1.md'}")


if __name__ == "__main__":
    main()
