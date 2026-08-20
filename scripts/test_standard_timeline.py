from __future__ import annotations

import json
import unittest
from pathlib import Path

from standard_timeline import find_version_topics, flatten_topic_text, get_version_topic, list_version_topics, load_standard_timeline, resolve_version_question


ROOT = Path(__file__).resolve().parents[1]
EVAL_FILE = ROOT / "data" / "eval_sets" / "fire_code_version_timeline.json"


class StandardTimelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.timeline = load_standard_timeline()

    def test_topics_and_versions_are_complete(self) -> None:
        self.assertGreaterEqual(len(self.timeline["topics"]), 3)
        for topic in self.timeline["topics"]:
            versions = topic.get("versions") or []
            self.assertGreaterEqual(len(versions), 2, topic["topic_id"])
            for version in versions:
                self.assertTrue(version.get("standard_id"))
                self.assertTrue(version.get("article_no"))
                self.assertTrue(version.get("effective_date"))
                self.assertTrue(version.get("requirement"))

    def test_dynamic_topics_are_generated_from_policy_files(self) -> None:
        payload = list_version_topics()
        topics = payload["topics"]
        kinds = {item.get("topic_kind") for item in topics}
        self.assertIn("generated", kinds)
        self.assertGreaterEqual(len(topics), len(self.timeline["topics"]) + 3)
        self.assertIn("standard_effective_timeline", {item["topic_id"] for item in topics})
        replacement_ids = [item["topic_id"] for item in topics if item["topic_id"].startswith("replacement_")]
        self.assertGreaterEqual(len(replacement_ids), 2)

    def test_versions_are_sorted_by_effective_date(self) -> None:
        for topic in self.timeline["topics"]:
            detail = get_version_topic(topic["topic_id"])
            dates = [item.get("effective_date", "") for item in detail["versions"]]
            self.assertEqual(dates, sorted(dates), topic["topic_id"])

    def test_eval_cases_are_covered_by_topic_text(self) -> None:
        payload = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            topic = get_version_topic(case["expected_topic_id"])
            text = flatten_topic_text(topic)
            for phrase in case["must_have"]:
                self.assertIn(phrase, text, f"{case['id']}: {phrase}")
            for value in case["expected_values"]:
                self.assertIn(value, text, f"{case['id']}: {value}")
            for standard_id in case["expected_standard_ids"]:
                self.assertIn(standard_id, text, f"{case['id']}: {standard_id}")

    def test_find_version_topics_by_keywords(self) -> None:
        hits = find_version_topics("新建住宅户门通行净宽是多少")
        self.assertEqual(hits[0]["topic_id"], "residential_entry_door_width")
        garage_hits = find_version_topics("地下汽车库设置自动灭火系统后疏散距离是多少")
        self.assertEqual(garage_hits[0]["topic_id"], "garage_egress_distance")

    def test_resolve_question_uses_matched_topic_without_retrieval(self) -> None:
        result = resolve_version_question("新建住宅户门通行净宽是多少")
        self.assertEqual(result["topic_id"], "residential_entry_door_width")
        self.assertEqual(result["matched_by"], "topic")

    def test_resolve_question_marks_versions_by_applicable_date(self) -> None:
        result = resolve_version_question("新建住宅户门通行净宽是多少", applicable_date="2024-01-01")
        versions = {item["standard_id"]: item for item in result["versions"]}
        self.assertTrue(versions["GB 55037-2022"]["applicable"])
        self.assertFalse(versions["GB 55038-2025"]["applicable"])


if __name__ == "__main__":
    unittest.main()
