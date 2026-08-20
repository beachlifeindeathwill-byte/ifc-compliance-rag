from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pdfplumber
from pypdf import PdfReader

from text_normalization import normalize_ocr_artifacts


ROOT = Path(__file__).resolve().parents[1]
RAW_PDF_DIR = ROOT / "data" / "raw_pdfs"
PROCESSED_DIR = ROOT / "data" / "processed_text"
AUDIT_DIR = ROOT / "data" / "chunks_audit"


@dataclass(frozen=True)
class StandardConfig:
    file_name: str
    standard_id: str
    title: str
    version: str
    priority: int
    max_chunk_chars: int
    overlap_chars: int
    notes: str = ""
    prefer_pypdf: bool = False


STANDARD_CONFIGS: dict[str, StandardConfig] = {
    "GB 50016-2014.pdf": StandardConfig(
        file_name="GB 50016-2014.pdf",
        standard_id="GB 50016-2014",
        title="建筑设计防火规范",
        version="2018年版",
        priority=30,
        max_chunk_chars=1200,
        overlap_chars=120,
        notes="长规范，表格多；优先保留条文和表格上下文。",
    ),
    "GB 50067-2014.pdf": StandardConfig(
        file_name="GB 50067-2014.pdf",
        standard_id="GB 50067-2014",
        title="汽车库、修车库、停车场设计防火规范",
        version="2014",
        priority=40,
        max_chunk_chars=900,
        overlap_chars=90,
        notes="短规范，条文密集；使用较小 chunk 提升精准召回。",
    ),
    "GB 55036-2022.pdf": StandardConfig(
        file_name="GB 55036-2022.pdf",
        standard_id="GB 55036-2022",
        title="消防设施通用规范",
        version="2022",
        priority=90,
        max_chunk_chars=1000,
        overlap_chars=100,
        notes="强制性消防设施通用规范；已替换为可提取文本版本，条文较短且表格少。",
    ),
    "GB 55037-2022.pdf": StandardConfig(
        file_name="GB 55037-2022.pdf",
        standard_id="GB 55037-2022",
        title="建筑防火通用规范",
        version="2022",
        priority=100,
        max_chunk_chars=1000,
        overlap_chars=100,
        notes="强制性通用规范；优先级高于旧版部分条文。",
    ),
    "GB 55038-2025.pdf": StandardConfig(
        file_name="GB 55038-2025.pdf",
        standard_id="GB 55038-2025",
        title="住宅项目规范",
        version="2025",
        priority=110,
        max_chunk_chars=900,
        overlap_chars=90,
        notes="现行强制性住宅项目规范；住宅设计与使用要求优先核验。",
    ),
    "GB 55038-2025_official_excerpt.pdf": StandardConfig(
        file_name="GB 55038-2025_official_excerpt.pdf",
        standard_id="GB 55038-2025",
        title="住宅项目规范（官方条文摘录）",
        version="2025",
        priority=110,
        max_chunk_chars=900,
        overlap_chars=90,
        notes="官方发布的可检索条文摘录；完整扫描原件单独留存用于人工复核。",
        prefer_pypdf=True,
    ),
}


ARTICLE_RE = re.compile(r"(?m)^\s*(\d+(?:\.\d+){1,3})\s+")
CONTINUATION_RE = re.compile(r"^\s*(?:[1-9]\d{0,1}\s+|[（(]?[一二三四五六七八九十1-9]\s*[）)]|续表|注[:：])")
CHAPTER_RE = re.compile(r"(?m)^\s*(第[一二三四五六七八九十百]+章|[1-9]\d?\s+[\u4e00-\u9fff].{0,30})\s*$")
TABLE_REF_RE = re.compile(r"(续表|表)\s*([0-9]+(?:\.[0-9]+)+(?:-\d+)?)")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = normalize_ocr_artifacts(text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = text.replace("\u3000", " ")
    text = text.replace("（", "(").replace("）", ")")
    text = re.sub(r"(?<=\d)\.\s+(?=\d)", ".", text)
    text = re.sub(r"(?<=\d)\s+(?=m\b|㎡|m2|人|层)", "", text)
    text = re.sub(r"[ \t]+", " ", text)

    cleaned_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.fullmatch(r"[•·\-\s]*\d+[•·\-\s]*", line):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def table_to_markdown(table: list[list[object]]) -> str:
    rows = []
    for row in table:
        cells = []
        for cell in row:
            value = "" if cell is None else str(cell)
            value = normalize_text(value).replace("\n", " ")
            cells.append(value)
        if any(cells):
            rows.append("| " + " | ".join(cells) + " |")
    if not rows:
        return ""
    return "[TABLE_START]\n" + "\n".join(rows) + "\n[TABLE_END]"


def extract_table_refs(text: str) -> list[tuple[str, str]]:
    return [(match.group(1), match.group(2)) for match in TABLE_REF_RE.finditer(text or "")]


def table_ref_from_text(text: str) -> tuple[str | None, bool]:
    refs = extract_table_refs(text)
    if not refs:
        return None, False
    label, table_no = refs[-1]
    return table_no, label == "续表"


def extract_table_context(page, bbox: tuple[float, float, float, float] | None) -> str:
    if not bbox:
        return ""
    try:
        _, top, _, bottom = bbox
        title_top = max(0, top - 110)
        title_bottom = min(float(page.height), bottom + 20)
        crop = page.crop((0, title_top, float(page.width), title_bottom))
        return normalize_text(crop.extract_text() or "")
    except Exception:
        return ""


def extract_table_blocks(page, page_text: str) -> list[dict]:
    blocks: list[dict] = []
    fallback_refs = extract_table_refs(page_text)
    try:
        tables = page.find_tables() or []
    except Exception:
        tables = []

    if tables:
        for idx, table_obj in enumerate(tables):
            try:
                raw_table = table_obj.extract()
            except Exception:
                raw_table = []
            markdown = table_to_markdown(raw_table)
            if not markdown:
                continue
            context = extract_table_context(page, getattr(table_obj, "bbox", None))
            table_no, is_continuation = table_ref_from_text(context)
            if not table_no and idx < len(fallback_refs):
                label, table_no = fallback_refs[idx]
                is_continuation = label == "续表"
            blocks.append(
                {
                    "table_index": idx,
                    "text": markdown,
                    "context": context,
                    "table_no": table_no,
                    "is_continuation": is_continuation,
                }
            )
        return blocks

    for idx, raw_table in enumerate(page.extract_tables() or []):
        markdown = table_to_markdown(raw_table)
        if not markdown:
            continue
        table_no = None
        is_continuation = False
        if idx < len(fallback_refs):
            label, table_no = fallback_refs[idx]
            is_continuation = label == "续表"
        blocks.append(
            {
                "table_index": idx,
                "text": markdown,
                "context": "",
                "table_no": table_no,
                "is_continuation": is_continuation,
            }
        )
    return blocks


def readable_ratio(text: str) -> float:
    nonspace = sum(1 for c in text if not c.isspace())
    if not nonspace:
        return 0.0
    readable = sum(
        1
        for c in text
        if "\u4e00" <= c <= "\u9fff" or (c.isascii() and (c.isalnum() or c in ".,;:!?()[]{}|/-_"))
    )
    return readable / nonspace


def choose_page_text(pdfplumber_text: str, pypdf_text: str) -> str:
    """Prefer the cleaner text layer when a PDF extractor includes repeated watermarks."""
    if not pypdf_text:
        return pdfplumber_text
    if not pdfplumber_text:
        return pypdf_text

    plumber_articles = len(ARTICLE_RE.findall(pdfplumber_text))
    pypdf_articles = len(ARTICLE_RE.findall(pypdf_text))
    watermark_inflation = len(pdfplumber_text) > max(600, len(pypdf_text) * 1.8)
    pypdf_keeps_structure = pypdf_articles >= plumber_articles or pypdf_articles >= 2
    if watermark_inflation and pypdf_keeps_structure:
        return pypdf_text
    return pdfplumber_text


def extract_pages(pdf_path: Path, config: StandardConfig) -> list[dict]:
    pages: list[dict] = []
    pypdf = PdfReader(str(pdf_path))
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            plumber_text = normalize_text(page.extract_text() or "")
            pypdf_text = normalize_text(pypdf.pages[page_index - 1].extract_text() or "")
            page_text = pypdf_text if config.prefer_pypdf and pypdf_text else choose_page_text(plumber_text, pypdf_text)
            table_blocks = extract_table_blocks(page, page_text)
            pages.append(
                {
                    "page": page_index,
                    "text": page_text,
                    "tables": table_blocks,
                    "char_count": len(page_text),
                    "table_count": len(table_blocks),
                    "article_count": len(ARTICLE_RE.findall(page_text)),
                    "readable_ratio": round(readable_ratio(page_text), 3),
                }
            )
    return pages


def find_chapter(text: str, previous: str | None) -> str | None:
    matches = CHAPTER_RE.findall(text)
    if matches:
        return matches[-1]
    return previous


def split_page_by_article(page_text: str) -> list[str]:
    matches = list(ARTICLE_RE.finditer(page_text))
    if not matches:
        return [page_text] if page_text.strip() else []

    segments = []
    if matches[0].start() > 0:
        prefix = page_text[: matches[0].start()].strip()
        if prefix:
            segments.append(prefix)
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(page_text)
        segment = page_text[start:end].strip()
        if segment:
            segments.append(segment)
    return segments


def split_long_text(text: str, max_chars: int, overlap_chars: int) -> Iterable[str]:
    if len(text) <= max_chars:
        yield text
        return

    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        window = text[start:end]
        cut = max(window.rfind("\n"), window.rfind("。"), window.rfind("；"))
        if cut > max_chars * 0.55:
            end = start + cut + 1
            window = text[start:end]
        yield window.strip()
        if end >= len(text):
            break
        start = max(0, end - overlap_chars)


def first_article_no(text: str) -> str | None:
    match = ARTICLE_RE.search(text)
    return match.group(1) if match else None


def looks_like_article_continuation(text: str) -> bool:
    return bool(CONTINUATION_RE.search((text or "").strip()[:120]))


def looks_like_orphaned_article_body(text: str, previous_text: str | None) -> bool:
    if not previous_text:
        return False
    previous = previous_text.strip()
    current = (text or "").strip()
    if not current:
        return False
    if first_article_no(current):
        return False
    if re.search(r"(?:不应|不得|应|宜)(?:小于|大于|低于|高于|超过|少于)?\s*$", previous):
        return bool(re.match(r"^(?:\d|[一二三四五六七八九十]+[、.)）])", current))
    if re.search(r"[:：]\s*$", previous):
        return bool(re.match(r"^(?:\d|[一二三四五六七八九十]+[、.)）])", current))
    if len(previous) <= 180 and not re.search(r"[。；;：:]$", previous):
        return True
    body_starters = ("安全出口", "疏散", "防火", "消防", "建筑", "房间", "场所", "道路", "门", "窗")
    return len(previous) <= 120 and current.startswith(body_starters)


def build_chunks(config: StandardConfig, pages: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    current_chapter = None
    current_article_no = None
    current_article_chunk_id = None
    current_article_text = None

    for page in pages:
        text = page["text"]
        if not text.strip():
            continue
        current_chapter = find_chapter(text, current_chapter)

        for segment in split_page_by_article(text):
            for part_index, part in enumerate(split_long_text(segment, config.max_chunk_chars, config.overlap_chars)):
                if not part:
                    continue
                article_no = first_article_no(part)
                inherited_article_no = False
                parent_chunk_id = None
                if article_no:
                    current_article_no = article_no
                elif current_article_no and looks_like_article_continuation(part):
                    article_no = current_article_no
                    inherited_article_no = True
                    parent_chunk_id = current_article_chunk_id
                elif current_article_no and looks_like_orphaned_article_body(part, current_article_text):
                    article_no = current_article_no
                    inherited_article_no = True
                    parent_chunk_id = current_article_chunk_id
                chunk_id = f"{config.standard_id.replace(' ', '_')}_p{page['page']:04d}_{len(chunks):05d}"
                chunk = {
                    "chunk_id": chunk_id,
                    "standard_id": config.standard_id,
                    "standard_title": config.title,
                    "version": config.version,
                    "priority": config.priority,
                    "source_file": config.file_name,
                    "page": page["page"],
                    "chapter": current_chapter,
                    "article_no": article_no,
                    "parent_article_no": article_no,
                    "parent_chunk_id": parent_chunk_id,
                    "inherited_article_no": inherited_article_no,
                    "has_table": "[TABLE_START]" in part,
                    "chunk_type": "article_or_table" if article_no or "[TABLE_START]" in part else "page_context",
                    "part_index": part_index,
                    "char_count": len(part),
                    "text": part,
                }
                chunks.append(chunk)
                if first_article_no(part):
                    current_article_chunk_id = chunk_id
                    current_article_text = part
        for table in page.get("tables") or []:
            table_text = table.get("text") or ""
            if not table_text:
                continue
            context = table.get("context") or ""
            table_no = table.get("table_no")
            article_no = table_no or first_article_no(context) or current_article_no
            inherited_article_no = bool(article_no and not first_article_no(table_text))
            chunk_id = f"{config.standard_id.replace(' ', '_')}_p{page['page']:04d}_{len(chunks):05d}_table{table.get('table_index', 0)}"
            parts = []
            if context:
                parts.append(context)
            if table_no:
                parts.append(f"表 {table_no}")
            parts.append(table_text)
            chunk = {
                "chunk_id": chunk_id,
                "standard_id": config.standard_id,
                "standard_title": config.title,
                "version": config.version,
                "priority": config.priority,
                "source_file": config.file_name,
                "page": page["page"],
                "chapter": current_chapter,
                "article_no": article_no,
                "parent_article_no": article_no,
                "parent_chunk_id": current_article_chunk_id if article_no == current_article_no else None,
                "inherited_article_no": inherited_article_no,
                "has_table": True,
                "table_no": table_no,
                "is_continuation_table": bool(table.get("is_continuation")),
                "chunk_type": "structured_table",
                "part_index": 0,
                "char_count": sum(len(part) for part in parts),
                "text": "\n\n".join(parts),
            }
            chunks.append(chunk)
    return chunks


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_audit_files(chunks: list[dict]) -> None:
    if AUDIT_DIR.exists():
        shutil.rmtree(AUDIT_DIR)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    for chunk in chunks:
        standard_dir = AUDIT_DIR / chunk["standard_id"].replace(" ", "_")
        standard_dir.mkdir(exist_ok=True)
        file_name = f"{chunk['chunk_id']}.txt"
        audit_text = (
            f"CHUNK_ID: {chunk['chunk_id']}\n"
            f"STANDARD: {chunk['standard_id']} {chunk['standard_title']}\n"
            f"VERSION: {chunk['version']}\n"
            f"SOURCE: {chunk['source_file']}\n"
            f"PAGE: {chunk['page']}\n"
            f"CHAPTER: {chunk['chapter']}\n"
            f"ARTICLE: {chunk['article_no']}\n"
            f"HAS_TABLE: {chunk['has_table']}\n"
            f"TYPE: {chunk['chunk_type']}\n"
            f"CHARS: {chunk['char_count']}\n"
            f"{'=' * 72}\n"
            f"{chunk['text']}\n"
        )
        (standard_dir / file_name).write_text(audit_text, encoding="utf-8", newline="\n")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks: list[dict] = []
    summaries: list[dict] = []
    skipped: list[dict] = []

    for pdf_path in sorted(RAW_PDF_DIR.glob("*.pdf")):
        config = STANDARD_CONFIGS.get(pdf_path.name)
        if not config:
            skipped.append({"file": pdf_path.name, "reason": "no_config"})
            continue

        print(f"Parsing {pdf_path.name} ...")
        pages = extract_pages(pdf_path, config)
        total_chars = sum(p["char_count"] for p in pages)
        if total_chars < 100:
            skipped.append(
                {
                    "file": pdf_path.name,
                    "standard_id": config.standard_id,
                    "reason": "no_extractable_text_needs_ocr",
                    "pages": len(pages),
                    "total_chars": total_chars,
                    "notes": config.notes,
                }
            )
            continue

        page_rows_path = PROCESSED_DIR / f"pages_{config.standard_id.replace(' ', '_')}.jsonl"
        write_jsonl(page_rows_path, pages)
        chunks = build_chunks(config, pages)
        all_chunks.extend(chunks)
        summaries.append(
            {
                "file": pdf_path.name,
                "standard_id": config.standard_id,
                "title": config.title,
                "version": config.version,
                "pages": len(pages),
                "total_chars": total_chars,
                "table_pages": sum(1 for p in pages if p["table_count"] > 0),
                "article_markers": sum(p["article_count"] for p in pages),
                "chunks": len(chunks),
                "chunks_with_article_no": sum(1 for c in chunks if c["article_no"]),
                "chunks_with_table": sum(1 for c in chunks if c["has_table"]),
                "max_chunk_chars": config.max_chunk_chars,
                "overlap_chars": config.overlap_chars,
                "notes": config.notes,
            }
        )

    write_jsonl(PROCESSED_DIR / "chunks.jsonl", all_chunks)
    write_audit_files(all_chunks)
    (PROCESSED_DIR / "chunk_summary.json").write_text(
        json.dumps({"summaries": summaries, "skipped": skipped}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (PROCESSED_DIR / "skipped_docs.json").write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Chunks: {len(all_chunks)}")
    for item in summaries:
        print(f"{item['standard_id']}: {item['chunks']} chunks, {item['chunks_with_article_no']} with article, {item['chunks_with_table']} with table")
    if skipped:
        print("Skipped:")
        for item in skipped:
            print(f"  {item['file']}: {item['reason']}")


if __name__ == "__main__":
    main()
