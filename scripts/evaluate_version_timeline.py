from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from standard_timeline import flatten_topic_text, get_version_topic, load_standard_timeline


ROOT = Path(__file__).resolve().parents[1]
EVAL_FILE = ROOT / "data" / "eval_sets" / "fire_code_version_timeline.json"
OUT_JSON = ROOT / "data" / "retrieval_runs" / "version_timeline_eval.json"
OUT_MD = ROOT / "data" / "retrieval_runs" / "version_timeline_eval.md"


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).lower()


def _check_expected_values(text: str, values: list[str]) -> list[str]:
    compact_text = _compact(text)
    missing: list[str] = []
    for value in values:
        if _compact(value) not in compact_text:
            missing.append(value)
    return missing


def evaluate_case(case: dict[str, Any], timeline: dict[str, Any]) -> dict[str, Any]:
    topic_id = case.get("expected_topic_id")
    topic = next((item for item in timeline.get("topics") or [] if item.get("topic_id") == topic_id), None)
    if not topic:
        return {
            **case,
            "passed": False,
            "errors": [f"missing topic {topic_id}"],
        }

    text = flatten_topic_text(get_version_topic(topic_id))
    compact_text = _compact(text)
    errors: list[str] = []

    for standard_id in case.get("expected_standard_ids") or []:
        if _compact(standard_id) not in compact_text:
            errors.append(f"missing standard {standard_id}")
    for value in _check_expected_values(text, case.get("expected_values") or []):
        errors.append(f"missing value {value}")
    for phrase in case.get("must_have") or []:
        if _compact(phrase) not in compact_text:
            errors.append(f"missing phrase {phrase}")
    if case.get("expect_change") and not topic.get("change_summary"):
        errors.append("expect_change is true but change_summary is empty")

    return {
        **case,
        "passed": not errors,
        "errors": errors,
    }


def write_report(results: list[dict[str, Any]], started_at: str) -> None:
    passed = sum(1 for item in results if item["passed"])
    total = len(results)
    report = {
        "status": "passed" if passed == total else "failed",
        "started_at": started_at,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0,
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    lines = [
        "# Version Timeline Evaluation",
        "",
        f"- Status: {report['status']}",
        f"- Total: {total}",
        f"- Passed: {passed}",
        f"- Failed: {total - passed}",
        f"- Pass Rate: {report['pass_rate']}",
        "",
    ]
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        lines.append(f"## {item['id']} {status}")
        lines.append(f"- Question: {item['question']}")
        lines.append(f"- Topic: {item['expected_topic_id']}")
        if item.get("errors"):
            lines.append(f"- Errors: {'; '.join(item['errors'])}")
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    payload = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    timeline = load_standard_timeline()
    results = [evaluate_case(case, timeline) for case in payload.get("cases") or []]
    write_report(results, started_at)
    passed = sum(1 for item in results if item["passed"])
    total = len(results)
    print(f"Version timeline eval: {passed}/{total} passed")
    for item in results:
        if not item["passed"]:
            print(f"  FAIL {item['id']}: {'; '.join(item['errors'])}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
