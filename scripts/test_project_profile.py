from __future__ import annotations

import unittest

from project_profile import apply_project_conflict_guard, classify_user_conditions, compose_compliance_context, normalize_project_profile


class ProjectProfileTests(unittest.TestCase):
    def test_normalizes_dynamic_questions_and_limits_count(self) -> None:
        raw = {
            "confirmed_conditions": ["多层住宅", "多层住宅"],
            "review_questions": [
                {"title": f"复核{i}", "question": f"问题{i}", "facts_used": ["模型事实"]}
                for i in range(8)
            ],
        }
        profile = normalize_project_profile(raw, "多层住宅")
        self.assertEqual(profile["confirmed_conditions"], ["多层住宅"])
        self.assertEqual(len(profile["review_questions"]), 6)
        self.assertEqual(profile["raw_project_context"], "多层住宅")

    def test_latest_user_context_has_explicit_priority_over_ifc_inference(self) -> None:
        context = compose_compliance_context(
            "请复核建筑分类",
            "多层住宅",
            {"建筑用途推断": {"value": "疑似住宅/居住建筑", "confidence": "low"}},
            {"confirmed_conditions": ["建筑使用功能为多层住宅"]},
        )
        self.assertIn("用户最新明确确认的项目条件 > IFC明确字段", context)
        self.assertIn("建筑使用功能为多层住宅", context)
        self.assertIn("不得再把该事实列为缺失", context)

    def test_uncertain_user_condition_is_not_promoted(self) -> None:
        profile = normalize_project_profile(
            {"confirmed_conditions": [], "uncertain_conditions": ["可能为住宅"]},
            "可能为住宅",
        )
        self.assertEqual(profile["confirmed_conditions"], [])
        self.assertEqual(profile["uncertain_conditions"], ["可能为住宅"])

    def test_mixed_free_text_conditions_are_classified_by_certainty_not_domain(self) -> None:
        confirmed, uncertain = classify_user_conditions("多层住宅；设置自动喷水灭火系统；耐火等级待确认")
        self.assertEqual(confirmed, ["多层住宅", "设置自动喷水灭火系统"])
        self.assertEqual(uncertain, ["耐火等级待确认"])

    def test_unresolved_project_fact_conflict_blocks_compliance_verdict(self) -> None:
        answer = {"verdict": "PASS", "can_answer": True, "certainty": "high", "missing_fields": []}
        result = apply_project_conflict_guard(
            answer,
            {"blocking_conflicts": ["用户报告门宽1.5m，IFC记录0.9m"]},
        )
        self.assertEqual(result["verdict"], "INSUFFICIENT_INFORMATION")
        self.assertEqual(result["verdict_adjusted_by"], "project_fact_conflict_guard")


if __name__ == "__main__":
    unittest.main()
