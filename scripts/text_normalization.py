from __future__ import annotations

import re


def normalize_ocr_artifacts(text: str) -> str:
    """Repair common OCR artifacts without changing the legal meaning."""
    if not text:
        return ""

    value = text
    replacements = {
        "不小子": "不小于",
        "不应小子": "不应小于",
        "小子": "小于",
        "大子": "大于",
        "不大子": "不大于",
        "不应大子": "不应大于",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)

    value = re.sub(r"(?<![A-Za-z])O\.\s*(\d+)", r"0.\1", value)
    value = re.sub(r"(?<![A-Za-z])o\.\s*(\d+)", r"0.\1", value)
    value = re.sub(r"(\d)\.\s*[Oo0]\s*[Oo0]\s*h\b", r"\1.00h", value, flags=re.IGNORECASE)
    value = re.sub(r"(\d)\.\s*[Oo0]\s*h\b", r"\1.0h", value, flags=re.IGNORECASE)
    value = re.sub(r"(\d+(?:\.\d+)?)\s*m[oO]\b", r"\1m", value)
    value = re.sub(r"(\d+(?:\.\d+)?)\s*[mM]\s*[’'`]", r"\1㎡", value)
    value = re.sub(r"(?<=\d)\s+(?=m\b|h\b|㎡|m2\b)", "", value, flags=re.IGNORECASE)
    return value


def normalize_for_retrieval(text: str) -> str:
    value = normalize_ocr_artifacts(text)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()
