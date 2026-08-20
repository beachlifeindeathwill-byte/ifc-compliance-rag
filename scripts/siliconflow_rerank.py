from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable


DEFAULT_RERANK_URL = "https://api.siliconflow.cn/v1/rerank"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True)
class RerankResult:
    index: int
    relevance_score: float
    document: str | None = None


class SiliconFlowRerankError(RuntimeError):
    pass


def rerank_documents(
    query: str,
    documents: Iterable[str],
    *,
    top_n: int = 8,
    model: str | None = None,
    api_key: str | None = None,
    return_documents: bool = False,
    timeout: int = 30,
) -> list[RerankResult]:
    docs = list(documents)
    if not query.strip():
        raise ValueError("query must not be empty")
    if not docs:
        return []

    key = api_key or os.getenv("SILICONFLOW_API_KEY")
    if not key:
        raise SiliconFlowRerankError("SILICONFLOW_API_KEY is not set")

    payload = {
        "model": model or os.getenv("SILICONFLOW_RERANK_MODEL", DEFAULT_RERANK_MODEL),
        "query": query,
        "documents": docs,
        "return_documents": return_documents,
        "top_n": min(max(1, top_n), len(docs)),
    }

    request = urllib.request.Request(
        os.getenv("SILICONFLOW_RERANK_URL", DEFAULT_RERANK_URL),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SiliconFlowRerankError(f"SiliconFlow rerank failed: HTTP {exc.code} {body}") from exc
    except urllib.error.URLError as exc:
        raise SiliconFlowRerankError(f"SiliconFlow rerank request failed: {exc.reason}") from exc

    data = json.loads(body)
    results = []
    for item in data.get("results", []):
        document = None
        if item.get("document"):
            document = item["document"].get("text")
        results.append(
            RerankResult(
                index=int(item["index"]),
                relevance_score=float(item.get("relevance_score", 0.0)),
                document=document,
            )
        )
    return results


def rerank_chunk_candidates(query: str, candidates: list[dict], *, top_n: int = 8) -> list[dict]:
    documents = [candidate["text"] for candidate in candidates]
    reranked = rerank_documents(query, documents, top_n=top_n)
    output = []
    for result in reranked:
        item = dict(candidates[result.index])
        item["rerank_score"] = result.relevance_score
        output.append(item)
    return output
