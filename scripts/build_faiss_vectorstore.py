from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import faiss
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "processed_text" / "chunks.jsonl"
VECTORSTORE_DIR = ROOT / "data" / "vectorstore"
DEFAULT_EMBEDDING_URL = "https://api.siliconflow.cn/v1/embeddings"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


def load_env_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"env file not found: {path}")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def embedding_input(chunk: dict) -> str:
    parts = [
        f"标准编号: {chunk.get('standard_id', '')}",
        f"标准名称: {chunk.get('standard_title', '')}",
        f"页码: {chunk.get('page', '')}",
    ]
    if chunk.get("article_no"):
        parts.append(f"条文号: {chunk['article_no']}")
    if chunk.get("chapter"):
        parts.append(f"章节: {chunk['chapter']}")
    parts.append(f"正文: {chunk.get('text', '')}")
    return "\n".join(parts)


def local_hash_embedding(text: str, *, dimension: int = 1024) -> list[float]:
    tokens = []
    compact = "".join(text.lower().split())
    tokens.extend(compact[i : i + 1] for i in range(len(compact)))
    tokens.extend(compact[i : i + 2] for i in range(max(0, len(compact) - 1)))
    tokens.extend(compact[i : i + 3] for i in range(max(0, len(compact) - 2)))

    vector = np.zeros(dimension, dtype="float32")
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        index = value % dimension
        sign = 1.0 if (value >> 10) & 1 else -1.0
        vector[index] += sign
    return vector.tolist()


def post_embeddings(inputs: list[str], *, api_key: str, model: str, url: str, timeout: int, retries: int) -> list[list[float]]:
    payload = {
        "model": model,
        "input": inputs,
        "encoding_format": "float",
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            items = sorted(data["data"], key=lambda item: item["index"])
            return [item["embedding"] for item in items]
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
        except urllib.error.URLError as exc:
            last_error = RuntimeError(str(exc.reason))

        sleep_seconds = min(2**attempt, 20)
        time.sleep(sleep_seconds)

    raise RuntimeError(f"embedding request failed after retries: {last_error}")


def embedding_cache_key(text: str, *, model: str) -> str:
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update(b"\0")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def load_embedding_cache(path: Path) -> dict[str, list[float]]:
    cache: dict[str, list[float]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            key = item.get("cache_key")
            if key:
                cache[str(key)] = item["embedding"]
    return cache


def append_embedding_cache(path: Path, items: list[tuple[str, int, list[float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for cache_key, row_id, embedding in items:
            f.write(json.dumps({"cache_key": cache_key, "row_id": row_id, "embedding": embedding}, ensure_ascii=False) + "\n")
        f.flush()


def build_embeddings(
    chunks: list[dict],
    *,
    api_key: str,
    model: str,
    url: str,
    batch_size: int,
    timeout: int,
    retries: int,
    cache_path: Path,
) -> np.ndarray:
    cache = load_embedding_cache(cache_path)
    if cache:
        print(f"loaded embedding cache {len(cache)}/{len(chunks)}")
    total = len(chunks)
    inputs_by_row_id = {row_id: embedding_input(chunk) for row_id, chunk in enumerate(chunks)}
    keys_by_row_id = {
        row_id: embedding_cache_key(input_text, model=model)
        for row_id, input_text in inputs_by_row_id.items()
    }
    for start in range(0, total, batch_size):
        row_ids = list(range(start, min(start + batch_size, total)))
        missing_ids = [row_id for row_id in row_ids if keys_by_row_id[row_id] not in cache]
        if not missing_ids:
            print(f"cached {min(start + batch_size, total)}/{total}")
            continue
        inputs = [inputs_by_row_id[row_id] for row_id in missing_ids]
        vectors = post_embeddings(inputs, api_key=api_key, model=model, url=url, timeout=timeout, retries=retries)
        cached_items = [(keys_by_row_id[row_id], row_id, vector) for row_id, vector in zip(missing_ids, vectors)]
        append_embedding_cache(cache_path, cached_items)
        for cache_key, _, vector in cached_items:
            cache[cache_key] = vector
        print(f"embedded {min(start + batch_size, total)}/{total}")

    missing = [row_id for row_id in range(total) if keys_by_row_id[row_id] not in cache]
    if missing:
        raise RuntimeError(f"embedding cache incomplete; missing {len(missing)} rows")
    array = np.asarray([cache[keys_by_row_id[row_id]] for row_id in range(total)], dtype="float32")
    if array.ndim != 2:
        raise RuntimeError(f"unexpected embedding shape: {array.shape}")
    return array


def build_local_hash_embeddings(chunks: list[dict], *, dimension: int) -> np.ndarray:
    vectors = []
    total = len(chunks)
    for row_id, chunk in enumerate(chunks, start=1):
        vectors.append(local_hash_embedding(embedding_input(chunk), dimension=dimension))
        if row_id % 250 == 0 or row_id == total:
            print(f"local embedded {row_id}/{total}")
    return np.asarray(vectors, dtype="float32")


def write_vectorstore(chunks: list[dict], embeddings: np.ndarray, *, provider: str, model: str, url: str | None) -> None:
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    fd, temp_index_path = tempfile.mkstemp(prefix="bim_fire_faiss_", suffix=".index")
    os.close(fd)
    try:
        faiss.write_index(index, temp_index_path)
        (VECTORSTORE_DIR / "faiss.index.tmp").write_bytes(Path(temp_index_path).read_bytes())
    finally:
        Path(temp_index_path).unlink(missing_ok=True)
    np.save(VECTORSTORE_DIR / "embeddings.npy.tmp", embeddings)

    with (VECTORSTORE_DIR / "chunk_meta.jsonl.tmp").open("w", encoding="utf-8", newline="\n") as f:
        for row_id, chunk in enumerate(chunks):
            meta = {k: v for k, v in chunk.items() if k != "text"}
            meta["row_id"] = row_id
            meta["text"] = chunk["text"]
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    config = {
        "version": "faiss_vectorstore_v1",
        "embedding_provider": provider,
        "embedding_model": model,
        "embedding_url": url,
        "index_type": "IndexFlatIP",
        "normalized": True,
        "metric": "cosine_similarity",
        "chunks": len(chunks),
        "dimension": int(embeddings.shape[1]),
        "source_chunks": str(CHUNKS_PATH.relative_to(ROOT)).replace("\\", "/"),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (VECTORSTORE_DIR / "embedding_config.json.tmp").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (VECTORSTORE_DIR / "faiss.index.tmp").replace(VECTORSTORE_DIR / "faiss.index")
    (VECTORSTORE_DIR / "embeddings.npy.tmp.npy").replace(VECTORSTORE_DIR / "embeddings.npy")
    (VECTORSTORE_DIR / "chunk_meta.jsonl.tmp").replace(VECTORSTORE_DIR / "chunk_meta.jsonl")
    (VECTORSTORE_DIR / "embedding_config.json.tmp").replace(VECTORSTORE_DIR / "embedding_config.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--model", default=os.getenv("SILICONFLOW_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--url", default=os.getenv("SILICONFLOW_EMBEDDING_URL", DEFAULT_EMBEDDING_URL))
    parser.add_argument("--provider", choices=["siliconflow", "local-hash"], default="siliconflow")
    parser.add_argument("--dimension", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="debug only; 0 means all chunks")
    parser.add_argument("--cache-path", type=Path, default=VECTORSTORE_DIR / "embedding_cache_v2.jsonl")
    args = parser.parse_args()

    if args.env_file:
        load_env_file(args.env_file)

    chunks = load_jsonl(CHUNKS_PATH)
    if args.limit:
        chunks = chunks[: args.limit]
    print(f"chunks {len(chunks)}")

    if args.provider == "siliconflow":
        api_key = os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            raise RuntimeError("SILICONFLOW_API_KEY is not set. Pass --env-file or set the environment variable.")
        embeddings = build_embeddings(
            chunks,
            api_key=api_key,
            model=args.model,
            url=args.url,
            batch_size=args.batch_size,
            timeout=args.timeout,
            retries=args.retries,
            cache_path=args.cache_path,
        )
        provider = "siliconflow"
        model = args.model
        url = args.url
    else:
        embeddings = build_local_hash_embeddings(chunks, dimension=args.dimension)
        provider = "local-hash"
        model = f"local-hash-{args.dimension}"
        url = None

    write_vectorstore(chunks, embeddings, provider=provider, model=model, url=url)
    print(f"wrote {VECTORSTORE_DIR / 'faiss.index'}")
    print(f"dimension {embeddings.shape[1]}")


if __name__ == "__main__":
    main()
