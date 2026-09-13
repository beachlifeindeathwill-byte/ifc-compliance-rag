from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "retrieval_runs"


def load_summary(path: Path) -> dict:
    if not path.exists():
        return {"missing": True}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload.get("summary"), dict):
        return payload["summary"]
    # IFC fact evaluation stores its summary fields at the report root.
    return {
        key: value
        for key, value in payload.items()
        if key not in {"results", "failures"}
    }


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temp, path)


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def write_live_report(report: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "eval_all_report.json"
    out_md = OUT_DIR / "eval_all_report.md"
    atomic_write(out_json, json.dumps(report, ensure_ascii=False, indent=2))

    lines = [
        "# Agent Evaluation Report",
        "",
        f"- Status: {report['status']}",
        f"- Started: {report['started_at']}",
        f"- Updated: {report['updated_at']}",
        f"- Finished: {report.get('finished_at') or 'running'}",
        f"- Revision: {report['revision']}",
        f"- Scope: {report.get('scope', 'all')}",
        f"- Note: `not_run` rows are cached historical summaries and were not executed in this report run.",
        "",
    ]
    for item in report["steps"]:
        lines.append(f"## {item['label']}")
        lines.append(f"- Status: {item['status']}")
        if item.get("started_at"):
            lines.append(f"- Started: {item['started_at']}")
        if item.get("finished_at"):
            lines.append(f"- Finished: {item['finished_at']}")
        if item.get("duration_seconds") is not None:
            lines.append(f"- Duration seconds: {item['duration_seconds']}")
        if item.get("exit_code") is not None:
            lines.append(f"- Exit code: {item['exit_code']}")
        for name, value in (item.get("summary") or {}).items():
            lines.append(f"- {name}: {json.dumps(value, ensure_ascii=False)}")
        lines.append("")
    atomic_write(out_md, "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--skip", nargs="*", default=[])
    args = parser.parse_args()

    all_steps = [
        {
            "key": "ifc",
            "label": "IFC Facts",
            "script": ROOT / "scripts" / "evaluate_ifc_facts.py",
            "out": OUT_DIR / "ifc_fact_eval.json",
            "extra": [],
        },
        {
            "key": "compliance",
            "label": "Compliance",
            "script": ROOT / "scripts" / "evaluate_compliance_cases.py",
            "out": OUT_DIR / "compliance_eval.json",
            "extra": ["--env-file", str(args.env_file)] if args.env_file else [],
        },
        {
            "key": "rag",
            "label": "RAG Focus",
            "script": ROOT / "scripts" / "evaluate_focus_tests.py",
            "out": OUT_DIR / "focus_single_turn_eval.json",
            "extra": ["--env-file", str(args.env_file)] if args.env_file else [],
        },
        {
            "key": "multiturn",
            "label": "Multi-Turn",
            "script": ROOT / "scripts" / "evaluate_multiturn_tests.py",
            "out": OUT_DIR / "multiturn_eval.json",
            "extra": ["--env-file", str(args.env_file), "--top-k", "8"] if args.env_file else ["--top-k", "8"],
        },
        {
            "key": "answers",
            "label": "RAG Answers",
            "script": ROOT / "scripts" / "evaluate_rag_answers.py",
            "out": OUT_DIR / "rag_answer_eval.json",
            "extra": ["--env-file", str(args.env_file)] if args.env_file else [],
        },
    ]

    steps = list(all_steps)
    if args.only:
        selected = set(args.only)
        steps = [step for step in steps if step["key"] in selected]
    if args.skip:
        skipped = set(args.skip)
        steps = [step for step in steps if step["key"] not in skipped]
    if not steps:
        print("No steps selected.")
        return

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    report = {
        "status": "running",
        "started_at": now,
        "updated_at": now,
        "finished_at": None,
        "revision": git_revision(),
        "scope": "all" if not args.only and not args.skip else ",".join(step["key"] for step in steps),
        "python": sys.version.split()[0],
        "steps": [
            {
                "key": step["key"],
                "label": step["label"],
                "status": "pending" if step in steps else "not_run",
                "output": str(step["out"].relative_to(ROOT)),
                "started_at": None,
                "finished_at": None,
                "duration_seconds": None,
                "exit_code": None,
                "summary": load_summary(step["out"]) if step not in steps else {},
                "output_updated_at": (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(step["out"].stat().st_mtime))
                    if step["out"].exists()
                    else None
                ),
            }
            for step in all_steps
        ],
        "summaries": {
            step["key"]: {
                "label": step["label"],
                "exit_code": None,
                "summary": load_summary(step["out"]),
                "status": "not_run",
            }
            for step in all_steps
            if step not in steps
        },
    }
    write_live_report(report)
    step_reports = {item["key"]: item for item in report["steps"]}
    for step in steps:
        step_report = step_reports[step["key"]]
        step_started = time.time()
        step_report["status"] = "running"
        step_report["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        report["updated_at"] = step_report["started_at"]
        write_live_report(report)
        command = [sys.executable, str(step["script"]), *step["extra"]]
        print(f"\n=== {step['label']} ===")
        result = subprocess.run(command, cwd=ROOT, env=None)
        step_report.update({
            "status": "passed" if result.returncode == 0 else "failed",
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(time.time() - step_started, 2),
            "exit_code": result.returncode,
            "summary": load_summary(step["out"]),
            "output_updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        report["summaries"][step["key"]] = {
            "label": step["label"],
            "exit_code": result.returncode,
            "summary": step_report["summary"],
            "status": step_report["status"],
        }
        report["updated_at"] = step_report["finished_at"]
        write_live_report(report)
        if result.returncode != 0:
            print(f"{step['label']} failed with exit code {result.returncode}")

    report["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["updated_at"] = report["finished_at"]
    selected_reports = [step_reports[step["key"]] for step in steps]
    selected_passed = all(item["status"] == "passed" for item in selected_reports)
    ran_all_steps = len(selected_reports) == len(all_steps)
    if selected_passed and ran_all_steps:
        report["status"] = "passed"
    elif selected_passed:
        report["status"] = "partial_passed"
    else:
        report["status"] = "failed"
    write_live_report(report)
    print(f"\nWrote {OUT_DIR / 'eval_all_report.md'}")


if __name__ == "__main__":
    main()
