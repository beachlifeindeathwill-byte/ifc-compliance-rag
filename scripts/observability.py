from __future__ import annotations

import threading
import time
from typing import Any


_lock = threading.Lock()
METRICS: dict[str, Any] = {
    "http": {
        "requests": 0,
        "failures": 0,
        "total_duration_ms": 0.0,
        "by_endpoint": {},
    },
    "llm": {
        "calls": 0,
        "failures": 0,
        "total_duration_ms": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    },
}


def record_http_request(path: str, duration_ms: float, ok: bool) -> None:
    with _lock:
        metrics = METRICS["http"]
        metrics["requests"] += 1
        metrics["total_duration_ms"] += duration_ms
        if not ok:
            metrics["failures"] += 1
        endpoint = metrics["by_endpoint"].setdefault(path, {"requests": 0, "duration_ms": 0.0})
        endpoint["requests"] += 1
        endpoint["duration_ms"] += duration_ms


def record_llm_usage(usage: dict[str, Any] | None, duration_ms: float, ok: bool = True) -> None:
    with _lock:
        metrics = METRICS["llm"]
        metrics["calls"] += 1
        metrics["total_duration_ms"] += duration_ms
        if not ok:
            metrics["failures"] += 1
        if usage:
            metrics["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            metrics["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            metrics["total_tokens"] += int(usage.get("total_tokens") or 0)


def snapshot_metrics() -> dict[str, Any]:
    with _lock:
        http = dict(METRICS["http"])
        http["by_endpoint"] = dict(http["by_endpoint"])
        return {"http": http, "llm": dict(METRICS["llm"])}


def now_ms() -> float:
    return time.perf_counter() * 1000.0
