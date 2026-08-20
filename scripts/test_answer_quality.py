from __future__ import annotations

import unittest

from answer_quality import answer_quality_snapshot, audit_atomic_claims


class AnswerQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [
            {
                "standard_id": "GB 55037-2022",
                "article_no": "7.1.4",
                "page": 38,
                "text": "疏散出口门的净宽度不应小于0.80m。",
            }
        ]

    def test_supported_requirement_passes(self) -> None:
        answer = {
            "can_answer": True,
            "certainty": "high",
            "citations": [1],
            "atomic_claims": [
                {
                    "text": "GB 55037-2022第7.1.4条规定疏散出口门净宽不应小于0.80m。",
                    "kind": "requirement",
                    "citations": [1],
                }
            ],
        }
        result = answer_quality_snapshot(answer, self.candidates)
        self.assertTrue(result["passed"])

    def test_unsupported_numeric_anchor_fails(self) -> None:
        answer = {
            "atomic_claims": [
                {"text": "疏散出口门净宽不应小于0.90m。", "kind": "requirement", "citations": [1]}
            ]
        }
        result = audit_atomic_claims(answer, self.candidates)
        self.assertFalse(result["passed"])
        self.assertIn("数值锚点", result["issues"][0])

    def test_project_fact_can_be_grounded_in_question(self) -> None:
        answer = {
            "atomic_claims": [
                {"text": "该门实际净宽为0.82m。", "kind": "project_fact", "citations": []}
            ]
        }
        result = audit_atomic_claims(answer, self.candidates, grounding_text="模型门净宽为0.82m")
        self.assertTrue(result["passed"])

    def test_invalid_citation_fails(self) -> None:
        answer = {
            "atomic_claims": [
                {"text": "疏散出口门有最低净宽要求。", "kind": "requirement", "citations": [2]}
            ]
        }
        result = audit_atomic_claims(answer, self.candidates)
        self.assertFalse(result["passed"])
        self.assertIn("引用编号越界", result["issues"][0])

    def test_missing_atomic_claims_do_not_break_legacy_answer(self) -> None:
        answer = {"can_answer": True, "certainty": "high", "citations": [1]}
        result = answer_quality_snapshot(answer, self.candidates)
        self.assertTrue(result["passed"])
        self.assertFalse(result["claim_audit"]["claims_present"])

    def test_table_header_unit_supports_numeric_cell(self) -> None:
        candidates = [
            {
                "standard_id": "GB 50067-2014",
                "article_no": "5.1.1",
                "text": "最大允许建筑面积(㎡)：地下汽车库、高层汽车库，一、二级2000",
            }
        ]
        answer = {
            "atomic_claims": [
                {"text": "地下汽车库基准值为2000㎡。", "kind": "requirement", "citations": [1]}
            ]
        }
        self.assertTrue(audit_atomic_claims(answer, candidates)["passed"])

    def test_verified_multiplication_supports_calculated_result(self) -> None:
        candidates = [
            {
                "standard_id": "GB 50067-2014",
                "article_no": "5.1.1",
                "text": "最大允许建筑面积(㎡)：地下汽车库一、二级2000；设置自动灭火系统为2.0倍。",
            }
        ]
        answer = {
            "atomic_claims": [
                {"text": "2000㎡×2.0=4000㎡。", "kind": "calculation", "citations": [1]}
            ]
        }
        self.assertTrue(audit_atomic_claims(answer, candidates)["passed"])

    def test_serialized_ifc_metric_field_supports_project_measurement(self) -> None:
        answer = {
            "atomic_claims": [
                {"text": "IFC模型中的门宽为1.25m。", "kind": "project_fact", "citations": []}
            ]
        }
        grounding = 'IFC模型事实：{"门":[{"width_m":1.25}]}'
        self.assertTrue(audit_atomic_claims(answer, [], grounding_text=grounding)["passed"])

    def test_equivalent_decimal_format_passes(self) -> None:
        candidates = [{"text": "室内疏散楼梯净宽度不应小于1.1m"}]
        answer = {
            "atomic_claims": [
                {"text": "室内疏散楼梯净宽度不应小于1.10m。", "kind": "requirement", "citations": [1]}
            ]
        }
        self.assertTrue(audit_atomic_claims(answer, candidates)["passed"])

    def test_result_before_parenthesized_formula_passes(self) -> None:
        candidates = [
            {"text": "最大允许建筑面积(㎡)地下汽车库2000；设置自动灭火系统为2.0倍。"}
        ]
        answer = {
            "atomic_claims": [
                {"text": "调整后为4000㎡（2000㎡×2.0）。", "kind": "calculation", "citations": [1]}
            ]
        }
        self.assertTrue(audit_atomic_claims(answer, candidates)["passed"])


if __name__ == "__main__":
    unittest.main()
