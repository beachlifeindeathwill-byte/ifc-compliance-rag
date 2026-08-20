from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from evaluate_retrieval import load_jsonl


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "processed_text" / "chunks.jsonl"
OUT_DIR = ROOT / "data" / "structured_tables"
TABLE_INDEX_PATH = OUT_DIR / "table_index.jsonl"

TABLE_REF_RE = re.compile(r"(续表|表)\s*([0-9]+(?:\.[0-9]+)+(?:-\d+)?)")
TABLE_BLOCK_RE = re.compile(r"\[TABLE_START\](.*?)\[TABLE_END\]", re.S)
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:m2|m²|㎡|m'|m|h|%|倍|层|人)?", re.I)
ADJUSTMENT_TERMS = (
    "增加",
    "减少",
    "放宽",
    "折减",
    "自动灭火",
    "自动喷水",
    "局部设置",
    "不应大于",
    "不应小于",
)


def clean_text(text: str, limit: int | None = None) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    return cleaned[:limit].strip() if limit else cleaned


def extract_table_refs(text: str) -> list[tuple[str, str]]:
    refs = []
    for match in TABLE_REF_RE.finditer(text):
        refs.append((match.group(1), match.group(2)))
    return refs


def nearest_table_no(text: str) -> tuple[str | None, bool]:
    refs = extract_table_refs(text)
    if not refs:
        return None, False
    label, no = refs[-1]
    return no, label == "续表"


def title_for_table(text: str, table_no: str | None) -> str:
    if not table_no:
        return ""
    compact = clean_text(text)
    match = re.search(rf"(?:续表|表)\s*{re.escape(table_no)}\s*([^。\[]{{0,80}})", compact)
    if not match:
        return ""
    return clean_text(match.group(0), 100)


def extract_table_blocks(text: str) -> list[str]:
    blocks = []
    for match in TABLE_BLOCK_RE.finditer(text):
        block = clean_text(match.group(0))
        if block:
            blocks.append(block)
    return blocks


def should_index_chunk(chunk: dict) -> bool:
    text = chunk.get("text", "")
    if "[TABLE_START]" in text:
        return True
    compact = clean_text(text)
    table_ref = TABLE_REF_RE.search(compact)
    if (
        table_ref
        and table_ref.start() <= 80
        and any(term in compact[:500] for term in ["最大", "最小", "允许", "面积", "净宽", "耐火", "防火分区"])
    ):
        return True
    return False


def table_reference_patterns(table_no: str) -> tuple[re.Pattern[str], ...]:
    escaped = re.escape(table_no)
    return (
        re.compile(rf"(?:表|续表)\s*{escaped}(?!\d)"),
        re.compile(rf"第\s*{escaped}\s*条"),
    )


def related_evidence_for_table(chunks: list[dict], table_chunk: dict, table_no: str | None) -> list[dict]:
    """Find separate clauses that modify or qualify a table's base values."""
    if not table_no:
        return []

    patterns = table_reference_patterns(table_no)
    related: list[dict] = []
    for chunk in chunks:
        if chunk.get("standard_id") != table_chunk.get("standard_id"):
            continue
        if chunk.get("chunk_id") == table_chunk.get("chunk_id"):
            continue
        text = chunk.get("text", "") or ""
        if not any(pattern.search(text) for pattern in patterns):
            continue
        matched_terms = [term for term in ADJUSTMENT_TERMS if term in text]
        if not matched_terms:
            continue
        related.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "article_no": chunk.get("article_no"),
                "page": chunk.get("page"),
                "matched_terms": matched_terms,
                "numbers": NUMBER_RE.findall(text),
                "text": text,
            }
        )
    return related


def table_family_metadata(chunks: list[dict]) -> dict[tuple[str, str], dict]:
    families: dict[tuple[str, str], dict] = {}
    for chunk in chunks:
        table_no = chunk.get("table_no")
        if not table_no:
            table_no, _ = nearest_table_no(chunk.get("text", ""))
        if not table_no or not should_index_chunk(chunk):
            continue
        key = (chunk.get("standard_id", ""), table_no)
        family = families.setdefault(key, {"chunk_ids": [], "pages": []})
        family["chunk_ids"].append(chunk.get("chunk_id"))
        if chunk.get("page") is not None:
            family["pages"].append(chunk.get("page"))
    for family in families.values():
        family["chunk_ids"] = list(dict.fromkeys(family["chunk_ids"]))
        family["pages"] = sorted(set(family["pages"]))
    return families


def build_entries(chunks: list[dict]) -> list[dict]:
    entries = []
    families = table_family_metadata(chunks)
    for chunk in chunks:
        if not should_index_chunk(chunk):
            continue
        text = chunk.get("text", "")
        table_no, is_continuation = nearest_table_no(text)
        table_no = chunk.get("table_no") or table_no
        is_continuation = bool(chunk.get("is_continuation_table")) or is_continuation
        blocks = extract_table_blocks(text)
        table_text = "\n\n".join(blocks) if blocks else text
        family = families.get((chunk.get("standard_id", ""), table_no), {})
        entry = {
            "table_id": f"{chunk['chunk_id']}_table",
            "chunk_id": chunk["chunk_id"],
            "standard_id": chunk["standard_id"],
            "standard_title": chunk.get("standard_title"),
            "source_file": chunk.get("source_file"),
            "page": chunk.get("page"),
            "article_no": chunk.get("article_no") or table_no,
            "table_no": table_no,
            "is_continuation": is_continuation,
            "title": title_for_table(text, table_no),
            "has_extracted_grid": bool(blocks),
            "family_chunk_ids": family.get("chunk_ids", []),
            "family_pages": family.get("pages", []),
            "related_evidence": related_evidence_for_table(chunks, chunk, table_no),
            "numbers": NUMBER_RE.findall(text),
            "text": table_text,
            "context_text": text,
        }
        entries.append(entry)
    return entries


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--out", type=Path, default=TABLE_INDEX_PATH)
    args = parser.parse_args()

    chunks = load_jsonl(args.chunks)
    entries = build_entries(chunks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, entries)

    summary = {
        "tables": len(entries),
        "with_table_no": sum(1 for item in entries if item.get("table_no")),
        "with_extracted_grid": sum(1 for item in entries if item.get("has_extracted_grid")),
        "standards": {},
    }
    for item in entries:
        standard = item["standard_id"]
        summary["standards"][standard] = summary["standards"].get(standard, 0) + 1
    (args.out.parent / "table_index_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
