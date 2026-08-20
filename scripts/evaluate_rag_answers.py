from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from api_server import QARequest, ask_qa
from build_faiss_vectorstore import load_env_file
from eval_progress import write_progress


DEFAULT_EVAL = ROOT / "data" / "eval_sets" / "rag_answer_cases.jsonl"
OUT_DIR = ROOT / "data" / "retrieval_runs"


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def compact(value: Any) -> str:
    return "".join(str(value or "").split()).lower()


def source_matches(sources: list[dict[str, Any]], standard_id: str | None, article: str | None) -> bool:
    if not standard_id and not article:
        return True
    for source in sources:
        if standard_id and source.get("standard_id") != standard_id:
            continue
        source_article = str(source.get("article_no") or "")
        if article and not (source_article == article or source_article.startswith(article + ".")):
            continue
        return True
    return False


def evaluate_case(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    answer = result.get("answer") or {}
    sources = result.get("sources") or []
    answer_text = compact(" ".join(str(answer.get(key) or "") for key in ("conclusion", "basis", "note")))
    assertions: list[dict[str, Any]] = []

    expected_can_answer = bool(case.get("expected_can_answer"))
    assertions.append(
        {
            "name": "can_answer",
            "passed": bool(answer.get("can_answer")) == expected_can_answer,
            "detail": f"expected={expected_can_answer} actual={bool(answer.get('can_answer'))}",
        }
    )
    for index, claim in enumerate(case.get("required_claims") or [], start=1):
        all_ok = all(compact(term) in answer_text for term in claim.get("all_of") or [])
        any_terms = claim.get("any_of") or []
        any_ok = not any_terms or any(compact(term) in answer_text for term in any_terms)
        source_ok = source_matches(sources, claim.get("standard_id"), claim.get("article"))
        assertions.append(
            {
                "name": f"required_claim_{index}",
                "passed": all_ok and any_ok and source_ok,
                "detail": f"all_of={all_ok} any_of={any_ok} source={source_ok}",
            }
        )
    for term in case.get("forbidden") or []:
        assertions.append(
            {
                "name": f"forbidden:{term}",
                "passed": compact(term) not in answer_text,
                "detail": "forbidden phrase must not occur",
            }
        )

    audit = answer.get("quality_audit") or {}
    audit_passed = bool(audit.get("passed"))
    assertions.append(
        {
            "name": "quality_audit",
            "passed": audit_passed,
            "detail": json.dumps(audit.get("structural_issues") or audit.get("claim_audit", {}).get("issues") or [], ensure_ascii=False),
        }
    )
    return {
        "id": case["id"],
        "question": case["question"],
        "passed": all(item["passed"] for item in assertions),
        "assertions": assertions,
        "can_answer": answer.get("can_answer"),
        "certainty": answer.get("certainty"),
        "conclusion": answer.get("conclusion"),
        "basis": answer.get("basis"),
        "quality_audit": audit,
        "sources": [
            {
                "standard_id": item.get("standard_id"),
                "article_no": item.get("article_no"),
                "page": item.get("page"),
            }
            for item in sources
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.env_file:
        load_env_file(args.env_file)

    cases = load_cases(args.eval_file)
    if args.limit:
        cases = cases[: args.limit]
    rows = []
    errors = 0
    write_progress("rag_answer_eval", status="running", completed=0, total=len(cases))
    for index, case in enumerate(cases, start=1):
        try:
            result = ask_qa(QARequest(question=case["question"], scope="auto", top_k=8, use_reranker=True))
            row = evaluate_case(case, result)
        except Exception as exc:
            errors += 1
            row = {"id": case["id"], "question": case["question"], "passed": False, "error": str(exc)}
        rows.append(row)
        print(f"evaluated {index}/{len(cases)} {case['id']} passed={row['passed']}")
        write_progress(
            "rag_answer_eval",
            status="running",
            completed=index,
            total=len(cases),
            current_id=case["id"],
            passed=sum(1 for item in rows if item["passed"]),
            errors=errors,
        )
        time.sleep(0.2)

    passed = sum(1 for row in rows if row["passed"])
    claim_assertions = [
        assertion
        for row in rows
        for assertion in row.get("assertions") or []
        if assertion["name"].startswith("required_claim_")
    ]
    summary = {
        "cases": len(rows),
        "passed": passed,
        "answer_pass_rate": round(passed / max(1, len(rows)), 3),
        "required_claim_pass_rate": round(
            sum(1 for item in claim_assertions if item["passed"]) / max(1, len(claim_assertions)), 3
        ),
        "quality_audit_pass_rate": round(
            sum(
                1
                for row in rows
                for item in row.get("assertions") or []
                if item["name"] == "quality_audit" and item["passed"]
            )
            / max(1, len(rows)),
            3,
        ),
        "errors": errors,
    }
    write_progress(
        "rag_answer_eval",
        status="passed" if errors == 0 and passed == len(rows) else "failed",
        completed=len(rows),
        total=len(rows),
        passed=passed,
        errors=errors,
        extra={"answer_pass_rate": summary["answer_pass_rate"]},
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "rag_answer_eval.json"
    out_md = OUT_DIR / "rag_answer_eval.md"
    out_json.write_text(json.dumps({"summary": summary, "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# RAG 回答质量评估",
        "",
        *[f"- {key}: {value}" for key, value in summary.items()],
        "",
        "## 用例",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"### {row['id']} {row['question']}",
                f"- Passed: {row['passed']}",
                f"- Conclusion: {row.get('conclusion')}",
                f"- Error: {row.get('error')}",
                "",
            ]
        )
    out_md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors or passed != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
