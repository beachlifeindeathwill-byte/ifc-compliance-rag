from __future__ import annotations

import importlib.util
import os
from typing import Any


def qdrant_enabled() -> bool:
    if not (os.getenv("QDRANT_URL") and os.getenv("QDRANT_COLLECTION")):
        return False
    return importlib.util.find_spec("qdrant_client") is not None


def qdrant_search_if_enabled(query_vector: Any, top_k: int) -> list[int] | None:
    if not qdrant_enabled():
        return None
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY") or None)
        results = client.query_points(
            collection_name=os.getenv("QDRANT_COLLECTION"),
            query=query_vector.tolist(),
            limit=top_k,
        )
        return [int(point.id) for point in results.points]
    except Exception:
        return None
