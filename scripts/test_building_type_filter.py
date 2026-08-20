from __future__ import annotations

import unittest

from evaluate_policy_retrieval import building_type_boost, detect_building_types


class BuildingTypeFilterTest(unittest.TestCase):
    def test_detect_building_types(self) -> None:
        self.assertEqual(detect_building_types("高层住宅户门净宽"), ["住宅"])
        self.assertEqual(detect_building_types("地下汽车库疏散距离"), ["汽车库"])

    def test_matching_building_type_gets_bonus(self) -> None:
        bonus, notes = building_type_boost("商业综合体疏散楼梯净宽", {"text": "商店营业厅的疏散楼梯净宽不应小于1.10m"})
        self.assertGreater(bonus, 0)
        self.assertTrue(any("命中建筑类型" in note for note in notes))

    def test_mismatched_building_type_gets_penalty(self) -> None:
        bonus, notes = building_type_boost("住宅户门净宽", {"text": "地下汽车库的疏散距离不应大于45m"})
        self.assertLess(bonus, 0)
        self.assertTrue(any("未命中建筑类型" in note for note in notes))


if __name__ == "__main__":
    unittest.main()
