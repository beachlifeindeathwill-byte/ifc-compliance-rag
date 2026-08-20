from __future__ import annotations

import unittest

from deterministic_compliance import apply_numeric_compliance_guard, evaluate_numeric_check
from llm_answer import normalize_compliance_answer


class DeterministicComplianceTests(unittest.TestCase):
    def test_blocks_pass_when_mixed_decision_has_unresolved_conditions(self) -> None:
        answer = {
            "verdict": "PASS",
            "can_answer": True,
            "certainty": "high",
            "numeric_checks": [
                {
                    "metric": "门净宽",
                    "actual_value": 0.9,
                    "actual_unit": "m",
                    "requirement_value": 0.8,
                    "requirement_unit": "m",
                    "comparator": "gte",
                    "expected_result": "PASS",
                }
            ],
            "citations": [],
            "decision_basis": "mixed",
            "blocking_conditions": ["门的实际用途未确认"],
            "non_numeric_failures": ["门的实际用途未确认"],
            "missing_fields": ["门的实际用途"],
        }
        result = apply_numeric_compliance_guard(answer, retrieval_question="门宽0.9m，限值0.8m")
        self.assertEqual(result["verdict"], "INSUFFICIENT_INFORMATION")
        self.assertEqual(result["verdict_adjusted_by"], "non_numeric_condition_guard")

    def test_keeps_pass_after_unsupported_supplemental_claims_were_excluded(self) -> None:
        answer = {
            "verdict": "PASS",
            "can_answer": True,
            "certainty": "medium",
            "numeric_checks": [
                {
                    "metric": "门净宽",
                    "actual_value": 0.9,
                    "actual_unit": "m",
                    "requirement_value": 0.8,
                    "requirement_unit": "m",
                    "comparator": "gte",
                    "expected_result": "PASS",
                }
            ],
            "citations": [],
            "decision_basis": "mixed",
            "blocking_conditions": [],
            "non_numeric_failures": ["已排除的补充结论"],
            "unsupported_claims_excluded": True,
        }
        result = apply_numeric_compliance_guard(answer, retrieval_question="门宽0.9m，限值0.8m")
        self.assertEqual(result["verdict"], "PASS")

    def test_advisory_risk_does_not_block_verified_pass(self) -> None:
        answer = {
            "verdict": "PASS",
            "can_answer": True,
            "certainty": "medium",
            "numeric_checks": [
                {
                    "metric": "楼梯净宽",
                    "actual_value": 1.35,
                    "actual_unit": "m",
                    "requirement_value": 1.3,
                    "requirement_unit": "m",
                    "comparator": "gte",
                    "expected_result": "PASS",
                }
            ],
            "citations": [],
            "decision_basis": "mixed",
            "blocking_conditions": [],
            "advisory_risks": ["表格OCR仍建议人工复核"],
            "non_numeric_failures": [],
        }
        result = apply_numeric_compliance_guard(answer, retrieval_question="楼梯净宽1.35m，限值1.30m")
        self.assertEqual(result["verdict"], "PASS")

    def test_evaluates_minimum_width(self) -> None:
        check = {
            "metric": "疏散出口门净宽",
            "actual_value": 1.1,
            "actual_unit": "m",
            "requirement_value": 0.8,
            "requirement_unit": "m",
            "comparator": "gte",
            "expected_result": "PASS",
        }
        result = evaluate_numeric_check(check, ["模型门宽 1.1m", "净宽不应小于0.80m"])
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["actual_result"], "PASS")
        self.assertTrue(result["source_matched"])

    def test_rejects_maximum_distance(self) -> None:
        check = {
            "metric": "疏散距离",
            "actual_value": 65,
            "actual_unit": "m",
            "requirement_value": 60,
            "requirement_unit": "m",
            "comparator": "lte",
            "expected_result": "PASS",
        }
        result = evaluate_numeric_check(check, ["疏散距离65m", "不应大于60m"])
        self.assertEqual(result["actual_result"], "FAIL")

    def test_converts_millimeters_to_meters(self) -> None:
        check = {
            "metric": "门净宽",
            "actual_value": 900,
            "actual_unit": "mm",
            "requirement_value": 0.8,
            "requirement_unit": "m",
            "comparator": "gte",
            "expected_result": "PASS",
        }
        result = evaluate_numeric_check(check, ["门宽900mm", "限值0.8m"])
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["actual_result"], "PASS")

    def test_rejects_incompatible_units(self) -> None:
        check = {
            "metric": "面积",
            "actual_value": 1800,
            "actual_unit": "m2",
            "requirement_value": 2,
            "requirement_unit": "h",
            "comparator": "lte",
            "expected_result": "PASS",
        }
        result = evaluate_numeric_check(check)
        self.assertEqual(result["status"], "unit_mismatch")

    def test_forces_fail_when_model_missed_numeric_violation(self) -> None:
        answer = {
            "verdict": "PASS",
            "can_answer": True,
            "certainty": "high",
            "citations": [1],
            "compliance_reasons": ["模型与规范一致"],
            "decision_basis": "numeric_only",
            "non_numeric_failures": [],
            "numeric_checks": [
                {
                    "citation": 1,
                    "metric": "疏散距离",
                    "actual_value": 65,
                    "actual_unit": "m",
                    "requirement_value": 60,
                    "requirement_unit": "m",
                    "comparator": "lte",
                    "expected_result": "PASS",
                }
            ],
        }
        result = apply_numeric_compliance_guard(
            answer,
            retrieval_question="地下汽车库疏散距离65m",
            candidates=[{"text": "设置自动灭火系统时不应大于60m"}],
        )
        self.assertEqual(result["verdict"], "FAIL")
        self.assertEqual(result["verdict_adjusted_by"], "deterministic_numeric_check")
        self.assertTrue(result["critical_risk"])

    def test_corrects_arithmetic_error_only_with_source_corroboration(self) -> None:
        answer = {
            "verdict": "FAIL",
            "can_answer": True,
            "certainty": "high",
            "citations": [1],
            "compliance_reasons": ["面积超过限值"],
            "decision_basis": "numeric_only",
            "non_numeric_failures": [],
            "numeric_checks": [
                {
                    "citation": 1,
                    "metric": "防火分区面积",
                    "actual_value": 1800,
                    "actual_unit": "m2",
                    "requirement_value": 2000,
                    "requirement_unit": "m2",
                    "comparator": "lte",
                    "expected_result": "FAIL",
                }
            ],
        }
        result = apply_numeric_compliance_guard(
            answer,
            retrieval_question="地下汽车库防火分区面积1800㎡",
            candidates=[{"text": "地下汽车库最大允许面积2000㎡"}],
        )
        self.assertEqual(result["verdict"], "PASS")

    def test_does_not_auto_pass_without_source_corroboration(self) -> None:
        answer = {
            "verdict": "FAIL",
            "can_answer": True,
            "certainty": "high",
            "citations": [1],
            "compliance_reasons": ["面积超过限值"],
            "decision_basis": "numeric_only",
            "non_numeric_failures": [],
            "numeric_checks": [
                {
                    "citation": 1,
                    "metric": "防火分区面积",
                    "actual_value": 1800,
                    "actual_unit": "m2",
                    "requirement_value": 2000,
                    "requirement_unit": "m2",
                    "comparator": "lte",
                    "expected_result": "FAIL",
                }
            ],
        }
        result = apply_numeric_compliance_guard(
            answer,
            retrieval_question="防火分区面积待复核",
            candidates=[{"text": "相关面积条文"}],
        )
        self.assertEqual(result["verdict"], "FAIL")

    def test_downgrades_pass_when_numeric_value_missing(self) -> None:
        answer = {
            "verdict": "PASS",
            "can_answer": True,
            "certainty": "high",
            "citations": [1],
            "compliance_reasons": ["门宽满足要求"],
            "decision_basis": "numeric_only",
            "non_numeric_failures": [],
            "numeric_checks": [
                {
                    "citation": 1,
                    "metric": "门净宽",
                    "actual_value": None,
                    "actual_unit": "m",
                    "requirement_value": 0.8,
                    "requirement_unit": "m",
                    "comparator": "gte",
                    "expected_result": "PASS",
                }
            ],
        }
        result = apply_numeric_compliance_guard(answer)
        self.assertEqual(result["verdict"], "INSUFFICIENT_INFORMATION")

    def test_replaces_contradictory_pass_conclusion(self) -> None:
        answer = {
            "verdict": "PASS",
            "can_answer": True,
            "certainty": "medium",
            "conclusion": "面积超过限值，但未超过规范允许值。",
            "compliance_reasons": ["数值比较"],
            "missing_fields": [],
            "critical_risk": False,
        }
        result = normalize_compliance_answer(answer)
        self.assertIn("满足规范限值", result["conclusion"])

    def test_replaces_contradictory_fail_conclusion(self) -> None:
        answer = {
            "verdict": "FAIL",
            "can_answer": True,
            "certainty": "medium",
            "conclusion": "该面积满足规范要求。",
            "compliance_reasons": ["数值比较"],
            "missing_fields": [],
            "critical_risk": False,
        }
        result = normalize_compliance_answer(answer)
        self.assertIn("不满足规范限值", result["conclusion"])


if __name__ == "__main__":
    unittest.main()
