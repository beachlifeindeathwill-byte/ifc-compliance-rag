from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import ifcopenshell


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IFC = ROOT / "data" / "ifc_uploads" / "8355749520aa_Duplex_A_20110907.ifc"
OUT_PATH = ROOT / "data" / "eval_sets" / "ifc_fact_cases.jsonl"


def _safe(value) -> str:
    return str(value or "").strip()


def _float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _storey_map(model) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for rel in model.by_type("IfcRelContainedInSpatialStructure"):
        structure = rel.RelatingStructure
        if not structure or not structure.is_a("IfcBuildingStorey"):
            continue
        storey_name = _safe(getattr(structure, "Name", None))
        for element in rel.RelatedElements:
            global_id = getattr(element, "GlobalId", None)
            if global_id:
                mapping[global_id] = storey_name
    return mapping


def _storey_elevations(model) -> list[dict[str, float | str]]:
    rows = []
    for storey in model.by_type("IfcBuildingStorey"):
        rows.append(
            {
                "name": _safe(getattr(storey, "Name", None)),
                "elevation_m": _float(getattr(storey, "Elevation", None)),
            }
        )
    rows = [row for row in rows if row["elevation_m"] is not None]
    rows.sort(key=lambda row: float(row["elevation_m"]))
    return rows


def _door_rows(model, storey_map: dict[str, str]) -> list[dict]:
    rows = []
    for door in model.by_type("IfcDoor"):
        global_id = getattr(door, "GlobalId", None)
        rows.append(
            {
                "global_id": _safe(global_id),
                "name": _safe(getattr(door, "Name", None)),
                "storey": storey_map.get(global_id, ""),
                "width_m": _float(getattr(door, "OverallWidth", None)),
                "height_m": _float(getattr(door, "OverallHeight", None)),
            }
        )
    return rows


def _space_rows(model, storey_map: dict[str, str]) -> list[dict]:
    rows = []
    for space in model.by_type("IfcSpace"):
        global_id = getattr(space, "GlobalId", None)
        rows.append(
            {
                "global_id": _safe(global_id),
                "name": _safe(getattr(space, "Name", None)) or _safe(getattr(space, "LongName", None)),
                "storey": storey_map.get(global_id, ""),
            }
        )
    return rows


def _count_types(model, types: list[str]) -> dict[str, int]:
    return {ifc_type: len(model.by_type(ifc_type)) for ifc_type in types}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ifc", type=Path, default=DEFAULT_IFC)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    model = ifcopenshell.open(str(args.ifc))
    storey_map = _storey_map(model)
    storeys = _storey_elevations(model)
    doors = _door_rows(model, storey_map)
    spaces = _space_rows(model, storey_map)
    types = [
        "IfcBuildingStorey",
        "IfcSpace",
        "IfcDoor",
        "IfcWindow",
        "IfcStair",
        "IfcRamp",
        "IfcWall",
        "IfcSlab",
        "IfcTransportElement",
    ]
    counts = _count_types(model, types)

    cases: list[dict] = []
    for ifc_type in types:
        cases.append(
            {
                "id": f"IFC_ENTITY_{ifc_type}",
                "layer": "entity",
                "question": f"该模型中有多少个 {ifc_type}？",
                "kind": "count",
                "target": ifc_type,
                "expected": counts[ifc_type],
            }
        )

    door_widths = sorted([row["width_m"] for row in doors if row["width_m"] is not None])
    door_heights = sorted([row["height_m"] for row in doors if row["height_m"] is not None])
    cases.extend(
        [
            {
                "id": "IFC_PROP_DOOR_WIDTHS",
                "layer": "property",
                "question": "模型中所有门宽是多少？",
                "kind": "numeric_list",
                "source": "doors",
                "field": "width_m",
                "expected": door_widths,
                "tolerance": 0.005,
            },
            {
                "id": "IFC_PROP_DOOR_HEIGHTS",
                "layer": "property",
                "question": "模型中所有门高是多少？",
                "kind": "numeric_list",
                "source": "doors",
                "field": "height_m",
                "expected": door_heights,
                "tolerance": 0.005,
            },
            {
                "id": "IFC_PROP_NARROW_DOORS",
                "layer": "property",
                "question": "门宽不大于 0.8m 的门有几扇？",
                "kind": "count_where",
                "source": "doors",
                "field": "width_m",
                "condition": "lte",
                "threshold": 0.8,
                "expected": sum(1 for width in door_widths if width <= 0.8),
            },
            {
                "id": "IFC_PROP_DOORS_WITH_WIDTH",
                "layer": "property",
                "question": "有明确门宽的 IfcDoor 有几扇？",
                "kind": "coverage",
                "source": "doors",
                "field": "width_m",
                "expected": len(door_widths),
            },
        ]
    )

    expected_door_storey = sorted([f"{row['storey']}|{row['global_id']}" for row in doors if row["storey"]])
    cases.append(
        {
            "id": "IFC_REL_DOOR_STOREY",
            "layer": "relation",
            "question": "每扇门分别属于哪个楼层？",
            "kind": "mapping",
            "source": "doors",
            "key_field": "global_id",
            "value_field": "storey",
            "expected": expected_door_storey,
        }
    )

    expected_space_storey = sorted([f"{row['storey']}|{row['global_id']}" for row in spaces if row["storey"]])
    cases.append(
        {
            "id": "IFC_REL_SPACE_STOREY",
            "layer": "relation",
            "question": "每个空间分别属于哪个楼层？",
            "kind": "mapping",
            "source": "spaces",
            "key_field": "global_id",
            "value_field": "storey",
            "expected": expected_space_storey,
        }
    )

    window_widths = sorted([_float(getattr(window, "OverallWidth", None)) for window in model.by_type("IfcWindow") if _float(getattr(window, "OverallWidth", None)) is not None])
    window_heights = sorted([_float(getattr(window, "OverallHeight", None)) for window in model.by_type("IfcWindow") if _float(getattr(window, "OverallHeight", None)) is not None])
    cases.extend(
        [
            {
                "id": "IFC_PROP_WINDOW_WIDTHS",
                "layer": "property",
                "question": "模型中所有窗宽是多少？",
                "kind": "numeric_list",
                "source": "windows",
                "field": "width_m",
                "expected": window_widths,
                "tolerance": 0.005,
            },
            {
                "id": "IFC_PROP_WINDOW_HEIGHTS",
                "layer": "property",
                "question": "模型中所有窗高是多少？",
                "kind": "numeric_list",
                "source": "windows",
                "field": "height_m",
                "expected": window_heights,
                "tolerance": 0.005,
            },
        ]
    )

    elevations = [float(row["elevation_m"]) for row in storeys]
    building_height = round(max(elevations) - min(elevations), 3) if len(elevations) > 1 else None
    intervals = [round(elevations[i + 1] - elevations[i], 3) for i in range(len(elevations) - 1)]
    average_interval = round(sum(intervals) / len(intervals), 3) if intervals else None
    cases.extend(
        [
            {
                "id": "IFC_DERIVED_BUILDING_HEIGHT",
                "layer": "derived",
                "question": "由楼层高程推算的建筑高度是多少？",
                "kind": "scalar",
                "source": "building_height_m",
                "expected": building_height,
                "tolerance": 0.01,
            },
            {
                "id": "IFC_DERIVED_AVG_STOREY_HEIGHT",
                "layer": "derived",
                "question": "相邻楼层平均高度是多少？",
                "kind": "nested_scalar",
                "source": ["storey_analysis", "average_interval_m"],
                "expected": average_interval,
                "tolerance": 0.01,
            },
            {
                "id": "IFC_DERIVED_STOREY_ELEVATIONS",
                "layer": "derived",
                "question": "各楼层高程是多少？",
                "kind": "numeric_list",
                "source": "storeys",
                "field": "elevation_m",
                "expected": sorted(elevations),
                "tolerance": 0.01,
            },
        ]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(json.dumps(
        {
            "file": args.ifc.name,
            "cases": len(cases),
            "counts": counts,
            "building_height_m": building_height,
            "average_storey_interval_m": average_interval,
        },
        ensure_ascii=False,
        indent=2,
    ))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
