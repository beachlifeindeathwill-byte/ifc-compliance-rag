from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "retrieval_runs"


def write_progress(
    name: str,
    *,
    status: str,
    completed: int,
    total: int,
    current_id: str | None = None,
    passed: int = 0,
    errors: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "name": name,
        "status": status,
        "completed": completed,
        "total": total,
        "progress": round(completed / max(1, total), 3),
        "current_id": current_id,
        "passed": passed,
        "errors": errors,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **(extra or {}),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}_progress.json"
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)
