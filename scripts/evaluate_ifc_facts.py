from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
DEFAULT_IFC = ROOT / "data" / "ifc_uploads" / "8355749520aa_Duplex_A_20110907.ifc"
DEFAULT_EVAL = ROOT / "data" / "eval_sets" / "ifc_fact_cases.jsonl"
OUT_DIR = ROOT / "data" / "retrieval_runs"

import sys

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ifc_fire_parser import parse_ifc_file


def _get_path(source, path: list[str] | str):
    if isinstance(path, str):
        return source.get(path)
    current = source
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _numeric_close(actual, expected, tolerance: float) -> bool:
    if actual is None or expected is None:
        return actual == expected
    try:
        return abs(float(actual) - float(expected)) <= float(tolerance)
    except Exception:
        return False


def _match_numeric_lists(actual, expected, tolerance: float) -> tuple[bool, float]:
    if not isinstance(actual, list) or not isinstance(expected, list):
        return False, 0.0
    actual = sorted(actual)
    expected = sorted(expected)
    if len(actual) != len(expected):
        return False, 0.0
    matched = sum(
        1 for left, right in zip(actual, expected) if _numeric_close(left, right, tolerance)
    )
    rate = matched / max(1, len(expected))
    return matched == len(expected), rate


def _set_f1(actual, expected) -> dict:
    actual = set(actual or [])
    expected = set(expected or [])
    intersection = actual & expected
    precision = len(intersection) / len(actual) if actual else (1.0 if not expected else 0.0)
    recall = len(intersection) / len(expected) if expected else (1.0 if not actual else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "actual_count": len(actual),
        "expected_count": len(expected),
        "missing": sorted(expected - actual)[:8],
        "extra": sorted(actual - expected)[:8],
    }


def _extract_actual(summary: dict, case: dict):
    kind = case["kind"]
    source = case.get("source")
    if kind == "count":
        return (summary.get("counts") or {}).get(case["target"])
    if kind == "numeric_list":
        items = summary.get(source) or []
        return [item.get(case["field"]) for item in items if item.get(case["field"]) is not None]
    if kind == "count_where":
        items = summary.get(source) or []
        values = [item.get(case["field"]) for item in items if item.get(case["field"]) is not None]
        threshold = float(case["threshold"])
        condition = case["condition"]
        if condition == "lte":
            return sum(1 for value in values if float(value) <= threshold)
        if condition == "gte":
            return sum(1 for value in values if float(value) >= threshold)
        return len(values)
    if kind == "coverage":
        items = summary.get(source) or []
        return sum(1 for item in items if item.get(case["field"]) is not None)
    if kind == "mapping":
        items = summary.get(source) or []
        return sorted(
            f"{item.get(case['value_field']) or ''}|{item.get(case['key_field']) or ''}"
            for item in items
            if item.get(case["value_field"])
        )
    if kind == "scalar":
        return _get_path(summary, source)
    if kind == "nested_scalar":
        return _get_path(summary, source)
    return None


def _evaluate_case(case: dict, actual) -> dict:
    kind = case["kind"]
    expected = case["expected"]
    if kind == "count" or kind == "coverage" or kind == "count_where":
        passed = actual == expected
        return {"passed": passed, "actual": actual, "expected": expected}
    if kind in {"scalar", "nested_scalar"}:
        passed = _numeric_close(actual, expected, case.get("tolerance", 0.0))
        return {
            "passed": passed,
            "actual": actual,
            "expected": expected,
            "tolerance": case.get("tolerance", 0.0),
        }
    if kind == "numeric_list":
        passed, rate = _match_numeric_lists(actual, expected, case.get("tolerance", 0.0))
        return {"passed": passed, "match_rate": round(rate, 3), "actual_count": len(actual or []), "expected_count": len(expected)}
    if kind == "mapping":
        return {"passed": actual == expected, "set_metrics": _set_f1(actual, expected)}
    return {"passed": False, "actual": actual, "expected": expected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ifc", type=Path, default=DEFAULT_IFC)
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cases = []
    with args.eval_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    if args.limit:
        cases = cases[: args.limit]

    summary = parse_ifc_file(str(args.ifc))
    rows = []
    layer_totals: Counter[str] = Counter()
    layer_passed: Counter[str] = Counter()
    overall_passed = 0

    for case in cases:
        actual = _extract_actual(summary, case)
        result = _evaluate_case(case, actual)
        result.update({"id": case["id"], "layer": case["layer"], "question": case["question"], "kind": case["kind"]})
        rows.append(result)
        layer_totals[case["layer"]] += 1
        if result["passed"]:
            layer_passed[case["layer"]] += 1
            overall_passed += 1

    total = len(cases)
    layers = sorted(layer_totals)
    layer_summary = {
        layer: {
            "cases": layer_totals[layer],
            "passed": layer_passed[layer],
            "accuracy": round(layer_passed[layer] / layer_totals[layer], 3),
        }
        for layer in layers
    }
    report = {
        "file": str(args.ifc),
        "cases": total,
        "overall_accuracy": round(overall_passed / max(1, total), 3),
        "layers": layer_summary,
        "failures": [
            row
            for row in rows
            if not row["passed"]
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "ifc_fact_eval.json"
    out_md = OUT_DIR / "ifc_fact_eval.md"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# IFC Fact Evaluation",
        "",
        f"- File: `{args.ifc}`",
        f"- Cases: {report['cases']}",
        f"- Overall accuracy: {report['overall_accuracy']}",
        "",
        "## Layers",
        "",
    ]
    for layer, metric in layer_summary.items():
        lines.append(f"- {layer}: {metric['passed']}/{metric['cases']} = {metric['accuracy']}")
    lines.extend(["", "## Failures", ""])
    for row in report["failures"]:
        lines.append(f"### {row['id']} {row['question']}")
        lines.append(f"- Layer: {row['layer']}, kind: {row['kind']}")
        lines.append(f"- Passed: {row['passed']}")
        if "set_metrics" in row:
            lines.append(f"- Set F1: {row['set_metrics']['f1']}, precision: {row['set_metrics']['precision']}, recall: {row['set_metrics']['recall']}")
            if row["set_metrics"]["missing"]:
                lines.append("- Missing: " + "、".join(row["set_metrics"]["missing"][:8]))
        else:
            lines.append(f"- Expected: {row.get('expected')}")
            lines.append(f"- Actual: {row.get('actual')}")
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
