from __future__ import annotations

import unittest

from answer_consistency import apply_answer_consistency_guard


class AnswerConsistencyTests(unittest.TestCase):
    def test_adds_missing_citation_for_mentioned_standard_article(self) -> None:
        answer = {
            "can_answer": True,
            "certainty": "high",
            "conclusion": "按 GB 10000-2020 第1.2.3条执行，同时核对 GB 20000-2021 第2.3.4条。",
            "citations": [1],
            "missing_fields": [],
        }
        candidates = [
            {"standard_id": "GB 10000-2020", "article_no": "1.2.3"},
            {"standard_id": "GB 20000-2021", "article_no": "2.3.4"},
        ]
        result = apply_answer_consistency_guard(answer, candidates)
        self.assertEqual(result["citations"], [1, 2])
        self.assertNotIn("consistency_issues", result)

    def test_blocks_standard_not_present_in_evidence(self) -> None:
        answer = {
            "can_answer": True,
            "certainty": "high",
            "conclusion": "依据 GB 30000-2022 第3.1.1条，限值为1.0m。",
            "citations": [1],
            "missing_fields": [],
        }
        candidates = [{"standard_id": "GB 10000-2020", "article_no": "1.2.3"}]
        result = apply_answer_consistency_guard(answer, candidates)
        self.assertFalse(result["can_answer"])
        self.assertEqual(result["certainty"], "low")
        self.assertIn("不在本次召回证据中", result["consistency_issues"][0])

    def test_downgrades_asserted_numeric_condition_declared_missing(self) -> None:
        answer = {
            "can_answer": True,
            "certainty": "high",
            "conclusion": "由于建筑高度大于18m，因此不适用例外条件。",
            "citations": [1],
            "missing_fields": ["建筑高度是否大于18m"],
        }
        candidates = [{"standard_id": "GB 10000-2020", "article_no": "1.2.3"}]
        result = apply_answer_consistency_guard(answer, candidates)
        self.assertTrue(result["can_answer"])
        self.assertEqual(result["certainty"], "medium")
        self.assertIn("不能据此给出唯一结论", result["conclusion"])

    def test_allows_conditional_rule_when_not_asserted_as_project_fact(self) -> None:
        answer = {
            "can_answer": True,
            "certainty": "medium",
            "conclusion": "建筑高度大于18m时按一般要求复核；不大于18m时还需核对例外条件。",
            "citations": [1],
            "missing_fields": ["建筑高度是否大于18m"],
        }
        candidates = [{"standard_id": "GB 10000-2020", "article_no": "1.2.3"}]
        result = apply_answer_consistency_guard(answer, candidates)
        self.assertNotIn("consistency_issues", result)

    def test_downgrades_compliance_verdict_on_fact_contradiction(self) -> None:
        answer = {
            "can_answer": True,
            "certainty": "high",
            "conclusion": "建筑高度超过24m，因此满足要求。",
            "citations": [1],
            "missing_fields": ["建筑高度是否超过24m"],
            "verdict": "PASS",
            "critical_risk": False,
        }
        candidates = [{"standard_id": "GB 10000-2020", "article_no": "1.2.3"}]
        result = apply_answer_consistency_guard(answer, candidates)
        self.assertEqual(result["verdict"], "INSUFFICIENT_INFORMATION")

    def test_blocks_causal_project_fact_not_provided_by_user(self) -> None:
        answer = {
            "can_answer": True,
            "certainty": "high",
            "conclusion": "楼梯按一般要求复核（因建筑高度大于18m，不适用例外）。",
            "citations": [1],
            "missing_fields": [],
        }
        candidates = [{"standard_id": "GB 10000-2020", "article_no": "1.2.3"}]
        result = apply_answer_consistency_guard(answer, candidates, grounding_text="建筑共4层")
        self.assertEqual(result["certainty"], "medium")
        self.assertIn("用户未提供", result["consistency_issues"][0])

    def test_allows_causal_project_fact_explicitly_provided_by_user(self) -> None:
        answer = {
            "can_answer": True,
            "certainty": "high",
            "conclusion": "因建筑高度大于18m，应按一般要求复核。",
            "citations": [1],
            "missing_fields": [],
        }
        candidates = [{"standard_id": "GB 10000-2020", "article_no": "1.2.3"}]
        result = apply_answer_consistency_guard(
            answer,
            candidates,
            grounding_text="项目建筑高度大于18m，共6层。",
        )
        self.assertNotIn("consistency_issues", result)

    def test_compliance_can_omit_unsupported_supplementary_claim(self) -> None:
        answer = {
            "can_answer": True,
            "certainty": "high",
            "conclusion": "门宽满足0.80m和0.90m两项要求。",
            "citations": [1],
            "missing_fields": [],
            "verdict": "PASS",
            "critical_risk": False,
            "atomic_claims": [
                {"text": "门宽不应小于0.80m。", "kind": "requirement", "citations": [1]},
                {"text": "门宽不应小于0.90m。", "kind": "requirement", "citations": [1]},
            ],
        }
        candidates = [{"text": "门宽不应小于0.80m。"}]
        result = apply_answer_consistency_guard(answer, candidates, allow_partial_claims=True)
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(len(result["atomic_claims"]), 1)
        self.assertEqual(len(result["omitted_atomic_claims"]), 1)


if __name__ == "__main__":
    unittest.main()
