from __future__ import annotations

import unittest

from ifc_fire_parser import _project_unit_factors, _space_status, _unit_quality


class FakeUnit:
    def __init__(self, prefix: str) -> None:
        self.Prefix = prefix
        self.UnitType = "LENGTHUNIT"

    def is_a(self, value: str) -> bool:
        return value == "IfcSIUnit"


class FakeProject:
    def __init__(self, prefix: str) -> None:
        self.UnitsInContext = type("Units", (), {"Units": [FakeUnit(prefix)]})()


class FakeModel:
    def __init__(self, prefix: str) -> None:
        self._projects = [FakeProject(prefix)]

    def by_type(self, value: str):
        return self._projects if value == "IfcProject" else []


class IFCUnitQualityTest(unittest.TestCase):
    def test_unit_quality_flags_negative_dimension(self) -> None:
        summary = {
            "doors": [{"name": "门1", "width_m": -1.0, "height_m": 2.0}],
            "spaces": [],
            "storeys": [],
        }
        result = _unit_quality(summary)
        self.assertTrue(result["reviewed"])
        self.assertTrue(any("非正数" in issue for issue in result["issues"]))

    def test_unit_quality_passes_normal_values(self) -> None:
        summary = {
            "doors": [{"name": "门1", "width_m": 1.2, "height_m": 2.1}],
            "spaces": [{"name": "空间1", "area": 18.5}],
            "storeys": [{"name": "Level 1", "elevation_m": 0.0}],
        }
        result = _unit_quality(summary)
        self.assertEqual(result["issues"], [])

    def test_project_unit_factors_reads_millimeter_unit(self) -> None:
        length_to_m, area_to_m2, label = _project_unit_factors(FakeModel("MILLI"))
        self.assertAlmostEqual(length_to_m, 0.001)
        self.assertAlmostEqual(area_to_m2, 0.000001)
        self.assertEqual(label, "mm")

    def test_space_status_is_neutral_when_ifc_has_no_spaces(self) -> None:
        result = _space_status([])
        self.assertEqual(result["status"], "not_defined")
        self.assertIn("不影响门、窗、楼梯", result["note"])

    def test_space_status_is_parsed_when_spaces_exist(self) -> None:
        result = _space_status([{"name": "空间1"}])
        self.assertEqual(result["status"], "parsed")
        self.assertEqual(result["count"], 1)


if __name__ == "__main__":
    unittest.main()
