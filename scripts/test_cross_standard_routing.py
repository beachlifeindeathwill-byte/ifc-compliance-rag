from __future__ import annotations

import unittest

from hybrid_retrieval import inject_routed_standard_evidence
from retrieval_policy import RouteDecision, infer_route, policy_bonus


class CrossStandardRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = {
            "standards": {
                "GB A-1": {
                    "title": "标准A",
                    "aliases": ["标准A"],
                    "scope_keywords": ["建筑防火"],
                    "priority": 100,
                    "route_bonus": 20,
                },
                "GB B-1": {
                    "title": "标准B",
                    "aliases": ["标准B"],
                    "scope_keywords": ["住宅户门", "新建住宅"],
                    "priority": 110,
                    "route_bonus": 30,
                },
            }
        }
        self.abolished = {"rules": []}

    def test_explicit_standard_keeps_scope_matched_companion(self) -> None:
        route = infer_route("按标准A，住宅户门净宽是多少", self.registry)
        self.assertEqual(route.primary, ["GB A-1"])
        self.assertEqual(route.secondary, ["GB B-1"])

    def test_secondary_scope_match_is_not_penalized_as_unrelated(self) -> None:
        route = RouteDecision(primary=["GB A-1"], secondary=["GB B-1"], reasons=[])
        bonus, notes = policy_bonus(
            "按标准A，住宅户门净宽是多少",
            {"standard_id": "GB B-1", "article_no": "1.0.1"},
            route,
            self.registry,
            self.abolished,
        )
        self.assertGreaterEqual(bonus, 4.0)
        self.assertNotIn("问题明确提到其他标准，当前标准降权", notes)

    def test_injects_missing_routed_standard_without_changing_top1(self) -> None:
        route = RouteDecision(primary=["GB A-1"], secondary=["GB B-1"], reasons=[])
        candidates = [
            {"chunk_id": f"a-{index}", "standard_id": "GB A-1", "hybrid_score": 100 - index}
            for index in range(6)
        ] + [{"chunk_id": "b-1", "standard_id": "GB B-1", "hybrid_score": 60}]

        result = inject_routed_standard_evidence(candidates, route, self.registry)
        self.assertEqual(result[0]["chunk_id"], "a-0")
        self.assertTrue(any(item["standard_id"] == "GB B-1" for item in result[:5]))
        self.assertEqual(len({item["chunk_id"] for item in result}), len(result))

if __name__ == "__main__":
    unittest.main()
