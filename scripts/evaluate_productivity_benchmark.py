from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "retrieval_runs"


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "通过"}


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            manual = float(raw.get("manual_seconds") or 0)
            assisted = float(raw.get("assisted_seconds") or 0)
            if manual <= 0 or assisted <= 0:
                continue
            rows.append(
                {
                    "task_id": raw.get("task_id", "").strip(),
                    "participant_role": raw.get("participant_role", "").strip(),
                    "manual_seconds": manual,
                    "assisted_seconds": assisted,
                    "manual_success": parse_bool(raw.get("manual_success", "")),
                    "assisted_success": parse_bool(raw.get("assisted_success", "")),
                    "notes": raw.get("notes", "").strip(),
                }
            )
    return rows


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"status": "no_data", "rows": 0}
    manual_total = sum(row["manual_seconds"] for row in rows)
    assisted_total = sum(row["assisted_seconds"] for row in rows)
    by_task: dict[str, dict] = {}
    for row in rows:
        task = by_task.setdefault(row["task_id"], {"rows": 0, "manual_seconds": 0.0, "assisted_seconds": 0.0})
        task["rows"] += 1
        task["manual_seconds"] += row["manual_seconds"]
        task["assisted_seconds"] += row["assisted_seconds"]
    for task in by_task.values():
        task["time_reduction_rate"] = 1 - task["assisted_seconds"] / task["manual_seconds"]
    return {
        "status": "passed",
        "rows": len(rows),
        "manual_total_seconds": round(manual_total, 2),
        "assisted_total_seconds": round(assisted_total, 2),
        "overall_time_reduction_rate": 1 - assisted_total / manual_total,
        "manual_success_rate": sum(1 for row in rows if row["manual_success"]) / len(rows),
        "assisted_success_rate": sum(1 for row in rows if row["assisted_success"]) / len(rows),
        "manual_p50_seconds": median(row["manual_seconds"] for row in rows),
        "assisted_p50_seconds": median(row["assisted_seconds"] for row in rows),
        "by_task": by_task,
    }


def write_report(summary: dict, out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    lines = [
        "# Productivity Benchmark Report",
        "",
        f"- Status: {summary.get('status')}",
        f"- Rows: {summary.get('rows', 0)}",
    ]
    if summary.get("status") == "passed":
        lines.extend(
            [
                f"- Overall time reduction: {pct(summary['overall_time_reduction_rate'])}",
                f"- Manual success rate: {pct(summary['manual_success_rate'])}",
                f"- Assisted success rate: {pct(summary['assisted_success_rate'])}",
                f"- Manual P50 seconds: {summary['manual_p50_seconds']}",
                f"- Assisted P50 seconds: {summary['assisted_p50_seconds']}",
                "",
                "## By Task",
            ]
        )
        for task_id, item in sorted(summary["by_task"].items()):
            lines.append(f"- {task_id}: {pct(item['time_reduction_rate'])} reduction, rows={item['rows']}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--out-json", type=Path, default=OUT_DIR / "productivity_benchmark.json")
    parser.add_argument("--out-md", type=Path, default=OUT_DIR / "productivity_benchmark.md")
    args = parser.parse_args()
    summary = summarize(load_rows(args.csv_path))
    write_report(summary, args.out_json, args.out_md)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
