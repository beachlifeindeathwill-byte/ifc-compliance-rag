from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from build_faiss_vectorstore import load_env_file
from api_server import IFCQuestionRequest, ask_ifc_compliance
from eval_progress import write_progress


DEFAULT_EVAL = ROOT / "data" / "eval_sets" / "compliance_cases.jsonl"
OUT_DIR = ROOT / "data" / "retrieval_runs"


def load_cases(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_case(case: dict, retries: int = 3) -> dict:
    payload = IFCQuestionRequest(
        summary=case["summary"],
        question=case["question"],
        project_context=case.get("project_context", ""),
        scope="auto",
        top_k=5,
        use_reranker=True,
    )
    last_error = None
    for attempt in range(retries):
        try:
            result = ask_ifc_compliance(payload)
            return result
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2)
    return {"error": f"compliance case failed after {retries} retries: {last_error}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", nargs="*", default=[])
    args = parser.parse_args()

    if args.env_file:
        load_env_file(args.env_file)

    cases = load_cases(args.eval_file)
    if args.limit:
        cases = cases[: args.limit]
    if args.ids:
        selected_ids = set(args.ids)
        cases = [case for case in cases if case["id"] in selected_ids]

    rows = []
    verdict_accuracy = 0
    expected_fail = 0
    safety_false_negative = 0
    error_count = 0
    verdict_counts: Counter[str] = Counter()
    write_progress("compliance_eval", status="running", completed=0, total=len(cases))

    for index, case in enumerate(cases, start=1):
        result = run_case(case)
        if result.get("error"):
            error_count += 1
            rows.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "expected_verdict": case["expected_verdict"],
                    "actual_verdict": "ERROR",
                    "passed": False,
                    "error": result["error"],
                    "conclusion": None,
                    "basis": None,
                    "compliance_reasons": [],
                    "missing_fields": [],
                    "quality_audit": {},
                    "answer_adjusted_by": None,
                    "numeric_verification": [],
                    "decision_basis": None,
                    "non_numeric_failures": [],
                    "top_sources": [],
                }
            )
            print(f"evaluated {index}/{len(cases)} {case['id']} error={result['error'][:120]}")
            write_progress(
                "compliance_eval",
                status="running",
                completed=index,
                total=len(cases),
                current_id=case["id"],
                passed=verdict_accuracy,
                errors=error_count,
            )
            continue
        actual_verdict = str(result.get("verdict") or result.get("answer", {}).get("verdict") or "INSUFFICIENT_INFORMATION")
        expected_verdict = case["expected_verdict"]
        passed = actual_verdict == expected_verdict
        if passed:
            verdict_accuracy += 1
        if expected_verdict == "FAIL":
            expected_fail += 1
            if actual_verdict == "PASS":
                safety_false_negative += 1
        verdict_counts[actual_verdict] += 1
        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected_verdict": expected_verdict,
                "actual_verdict": actual_verdict,
                "passed": passed,
                "reason": case.get("reason"),
                "conclusion": result.get("answer", {}).get("conclusion"),
                "basis": result.get("answer", {}).get("basis"),
                "compliance_reasons": result.get("answer", {}).get("compliance_reasons"),
                "missing_fields": result.get("answer", {}).get("missing_fields"),
                "quality_audit": result.get("answer", {}).get("quality_audit"),
                "answer_adjusted_by": result.get("answer", {}).get("answer_adjusted_by"),
                "numeric_verification": result.get("answer", {}).get("numeric_verification"),
                "decision_basis": result.get("answer", {}).get("decision_basis"),
                "non_numeric_failures": result.get("answer", {}).get("non_numeric_failures"),
                "top_sources": [
                    {
                        "standard_id": item.get("standard_id"),
                        "article_no": item.get("article_no"),
                        "page": item.get("page"),
                    }
                    for item in (result.get("sources") or [])[:3]
                ],
            }
        )
        print(f"evaluated {index}/{len(cases)} {case['id']} expected={expected_verdict} actual={actual_verdict} passed={passed}")
        write_progress(
            "compliance_eval",
            status="running",
            completed=index,
            total=len(cases),
            current_id=case["id"],
            passed=verdict_accuracy,
            errors=error_count,
        )

    total = len(cases)
    summary = {
        "cases": total,
        "verdict_accuracy": round(verdict_accuracy / max(1, total), 3),
        "verdict_counts": dict(verdict_counts),
        "expected_fail_cases": expected_fail,
        "safety_false_negative_rate": round(safety_false_negative / max(1, expected_fail), 3),
        "safety_false_negative_count": safety_false_negative,
        "error_count": error_count,
    }
    write_progress(
        "compliance_eval",
        status="passed" if error_count == 0 and verdict_accuracy == total else "failed",
        completed=total,
        total=total,
        passed=verdict_accuracy,
        errors=error_count,
        extra={"verdict_accuracy": summary["verdict_accuracy"]},
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if args.ids or args.limit:
        selected_ids = ",".join(args.ids) if args.ids else f"first_{args.limit}"
        suffix = f".ids.{selected_ids}"
    out_json = OUT_DIR / f"compliance_eval{suffix}.json"
    out_md = OUT_DIR / f"compliance_eval{suffix}.md"
    out_json.write_text(json.dumps({"summary": summary, "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Compliance Evaluation",
        "",
        f"- Cases: {summary['cases']}",
        f"- Verdict accuracy: {summary['verdict_accuracy']}",
        f"- Safety false negative rate: {summary['safety_false_negative_rate']}",
        f"- Safety false negative count: {summary['safety_false_negative_count']}",
        f"- Errors: {summary['error_count']}",
        f"- Verdict distribution: {json.dumps(summary['verdict_counts'], ensure_ascii=False)}",
        "",
        "## Cases",
        "",
    ]
    for row in rows:
        lines.append(f"### {row['id']} {row['question']}")
        lines.append(f"- Expected: {row['expected_verdict']} / Actual: {row['actual_verdict']} / Passed: {row['passed']}")
        lines.append(f"- Conclusion: {row.get('conclusion')}")
        lines.append(f"- Basis: {row.get('basis')}")
        missing_text = ", ".join(row.get("missing_fields") or []) or "none"
        lines.append(f"- Missing fields: {missing_text}")
        sources = "；".join([f"{item['standard_id']} {item['article_no']} p{item['page']}" for item in row["top_sources"]])
        lines.append(f"- Top sources: {sources}")
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {out_md}")
    if error_count or verdict_accuracy != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
