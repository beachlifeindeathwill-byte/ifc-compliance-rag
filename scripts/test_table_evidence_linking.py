from __future__ import annotations

import unittest

from build_table_index import build_entries
from table_retrieval import related_evidence_candidates


class TableEvidenceLinkingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            {
                "chunk_id": "table-base",
                "standard_id": "GB TEST-1",
                "standard_title": "测试规范",
                "source_file": "test.pdf",
                "page": 10,
                "article_no": "4.2.1",
                "table_no": "4.2.1",
                "has_table": True,
                "chunk_type": "structured_table",
                "text": "表 4.2.1 最大允许面积\n[TABLE_START]\n| 类别 | 面积 |\n| 地下 | 2000㎡ |\n[TABLE_END]",
            },
            {
                "chunk_id": "continuation",
                "standard_id": "GB TEST-1",
                "standard_title": "测试规范",
                "source_file": "test.pdf",
                "page": 11,
                "article_no": "4.2.1",
                "table_no": "4.2.1",
                "is_continuation_table": True,
                "has_table": True,
                "chunk_type": "structured_table",
                "text": "续表 4.2.1\n[TABLE_START]\n| 地上 | 3000㎡ |\n[TABLE_END]",
            },
            {
                "chunk_id": "modifier",
                "standard_id": "GB TEST-1",
                "standard_title": "测试规范",
                "source_file": "test.pdf",
                "page": 12,
                "article_no": "4.2.2",
                "has_table": False,
                "text": "4.2.2 设置自动灭火系统时，面积不应大于第4.2.1条规定的2.0倍。",
            },
            {
                "chunk_id": "unrelated-standard",
                "standard_id": "GB OTHER-1",
                "page": 12,
                "article_no": "4.2.2",
                "text": "设置自动灭火系统时，按第4.2.1条增加3.0倍。",
            },
        ]

    def test_builds_table_family_and_separate_related_clause(self) -> None:
        entries = build_entries(self.chunks)
        base = next(item for item in entries if item["chunk_id"] == "table-base")
        self.assertEqual(base["family_pages"], [10, 11])
        self.assertEqual(base["family_chunk_ids"], ["table-base", "continuation"])
        self.assertEqual([item["chunk_id"] for item in base["related_evidence"]], ["modifier"])

    def test_returns_modifier_only_when_query_requests_adjustment(self) -> None:
        tables = build_entries(self.chunks)
        for table in tables:
            table["table_score"] = 80.0
        candidates = related_evidence_candidates("地下设置自动灭火系统后面积增加到多少", tables)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["chunk_id"], "modifier")
        self.assertEqual(candidates[0]["article_no"], "4.2.2")

        without_adjustment = related_evidence_candidates("地下最大允许面积是多少", tables)
        self.assertEqual(without_adjustment, [])


if __name__ == "__main__":
    unittest.main()
