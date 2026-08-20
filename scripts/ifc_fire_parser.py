from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any


FIRE_KEYWORDS = (
    "fire",
    "消防",
    "防火",
    "sprinkler",
    "hydrant",
    "alarm",
    "smoke",
    "排烟",
    "防烟",
    "extinguisher",
    "灭火",
)

GARAGE_KEYWORDS = ("garage", "parking", "car park", "汽车库", "车库", "停车")
ATRIUM_KEYWORDS = ("atrium", "中庭")
STAIR_KEYWORDS = ("stair", "楼梯", "梯")
EXIT_KEYWORDS = ("exit", "egress", "出口", "疏散", "安全出口")
RESIDENTIAL_KEYWORDS = ("duplex", "residential", "residence", "house", "home", "apartment", "住宅", "住户", "卧室", "bedroom")
PUBLIC_KEYWORDS = ("office", "mall", "shop", "school", "hospital", "hotel", "公共", "商业", "办公", "学校", "医院", "旅馆")


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def _entity_name(entity: Any) -> str:
    parts = [
        _safe_str(getattr(entity, "Name", "")),
        _safe_str(getattr(entity, "LongName", "")),
        _safe_str(getattr(entity, "ObjectType", "")),
        _safe_str(getattr(entity, "Tag", "")),
    ]
    return " | ".join([part for part in parts if part])


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = _safe_str(value)
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def _project_unit_factors(model: Any) -> tuple[float, float, str]:
    length_to_m = 1.0
    length_label = "m"
    try:
        projects = model.by_type("IfcProject")
        if not projects:
            return length_to_m, length_to_m * length_to_m, length_label
        units = projects[0].UnitsInContext.Units
        for unit in units:
            if not unit.is_a("IfcSIUnit") or getattr(unit, "UnitType", None) != "LENGTHUNIT":
                continue
            prefix = str(getattr(unit, "Prefix", "") or "")
            factors = {
                "MILLI": 0.001,
                "CENTI": 0.01,
                "DECI": 0.1,
                "DECA": 10.0,
                "HECTO": 100.0,
                "KILO": 1000.0,
            }
            length_to_m = factors.get(prefix, 1.0)
            labels = {"MILLI": "mm", "CENTI": "cm", "DECI": "dm", "DECA": "dam", "HECTO": "hm", "KILO": "km"}
            length_label = labels.get(prefix, "m")
    except Exception:
        pass
    return length_to_m, length_to_m * length_to_m, length_label


def _load_get_psets():
    try:
        from ifcopenshell.util.element import get_psets

        return get_psets
    except Exception:
        return None


def _flatten_props(entity: Any, *, max_items: int = 80) -> dict[str, Any]:
    get_psets = _load_get_psets()
    if not get_psets:
        return {}
    try:
        psets = get_psets(entity) or {}
    except Exception:
        return {}
    flat: dict[str, Any] = {}
    for pset_name, values in psets.items():
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if key == "id" or isinstance(value, (dict, list, tuple)):
                continue
            flat[f"{pset_name}.{key}"] = value
            if len(flat) >= max_items:
                return flat
    return flat


def _first_prop(props: dict[str, Any], names: tuple[str, ...]) -> Any:
    normalized = {key.lower().replace(" ", ""): value for key, value in props.items()}
    for wanted in names:
        wanted_norm = wanted.lower().replace(" ", "")
        for key, value in normalized.items():
            if key.endswith(wanted_norm) or wanted_norm in key:
                return value
    return None


def _storey_name(entity: Any, storey_index: dict[str, str] | None = None) -> str | None:
    if storey_index:
        global_id = getattr(entity, "GlobalId", None)
        if global_id and _safe_str(global_id) in storey_index:
            return storey_index[_safe_str(global_id)]
    try:
        rels = getattr(entity, "ContainedInStructure", None) or []
        for rel in rels:
            structure = getattr(rel, "RelatingStructure", None)
            if structure:
                return _safe_str(getattr(structure, "Name", "")) or None
    except Exception:
        return None
    return None


def _build_storey_index(model: Any) -> dict[str, str]:
    index: dict[str, str] = {}
    try:
        rels = model.by_type("IfcRelContainedInSpatialStructure")
    except Exception:
        return index
    for rel in rels:
        structure = getattr(rel, "RelatingStructure", None)
        if not structure or structure.is_a() != "IfcBuildingStorey":
            continue
        storey_name = _safe_str(getattr(structure, "Name", "")) or "未命名楼层"
        for element in getattr(rel, "RelatedElements", None) or []:
            global_id = _safe_str(getattr(element, "GlobalId", ""))
            if global_id:
                index[global_id] = storey_name
    return index


def _has_any(text: str, keywords: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(keyword.lower() in lower for keyword in keywords)


def _short(text: str, limit: int = 80) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit]


def _display_name(text: str, *, fallback: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"\b\d{5,}\b", "", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s*mm\s*x\s*\d+(?:\.\d+)?\s*mm\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[:|_\\/-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:42] if text else fallback


def _sample_entities(model: Any, ifc_type: str, limit: int = 20) -> list[Any]:
    try:
        return list(model.by_type(ifc_type))[:limit]
    except Exception:
        return []


def _count_type(model: Any, ifc_type: str) -> int:
    try:
        return len(model.by_type(ifc_type))
    except Exception:
        return 0


def _project_info(model: Any) -> dict[str, Any]:
    project = (_sample_entities(model, "IfcProject", 1) or [None])[0]
    site = (_sample_entities(model, "IfcSite", 1) or [None])[0]
    building = (_sample_entities(model, "IfcBuilding", 1) or [None])[0]
    return {
        "schema": _safe_str(getattr(model, "schema", "")),
        "project_name": _safe_str(getattr(project, "Name", "")) if project else "",
        "site_name": _safe_str(getattr(site, "Name", "")) if site else "",
        "building_name": _safe_str(getattr(building, "Name", "")) if building else "",
    }


def _extract_storeys(model: Any, length_to_m: float = 1.0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entity in _sample_entities(model, "IfcBuildingStorey", 80):
        elevation = _as_float(getattr(entity, "Elevation", None))
        props = _flatten_props(entity, max_items=30)
        rows.append(
            {
                "name": _safe_str(getattr(entity, "Name", "")) or "未命名楼层",
                "elevation_m": round(elevation * length_to_m, 3) if elevation is not None else None,
                "fire_rating": _first_prop(props, ("FireRating", "FireResistanceRating", "耐火等级")),
                "raw_type": entity.is_a(),
            }
        )
    return rows


def _extract_spaces(model: Any, storey_index: dict[str, str], area_to_m2: float = 1.0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entity in _sample_entities(model, "IfcSpace", 180):
        props = _flatten_props(entity, max_items=80)
        name = _entity_name(entity)
        raw_area = _first_prop(
            props,
            (
                "NetFloorArea",
                "GrossFloorArea",
                "Area",
                "净面积",
                "建筑面积",
            ),
        )
        area_value = _as_float(raw_area)
        rows.append(
            {
                "global_id": _safe_str(getattr(entity, "GlobalId", "")),
                "name": _short(name) or "未命名空间",
                "storey": _storey_name(entity, storey_index),
                "area": round(area_value * area_to_m2, 3) if area_value is not None else None,
                "is_garage_related": _has_any(name, GARAGE_KEYWORDS),
                "is_atrium_related": _has_any(name, ATRIUM_KEYWORDS),
                "raw_type": entity.is_a(),
            }
        )
    return rows


def _extract_doors(model: Any, storey_index: dict[str, str], length_to_m: float = 1.0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entity in _sample_entities(model, "IfcDoor", 200):
        props = _flatten_props(entity, max_items=80)
        name = _entity_name(entity)
        width = _as_float(getattr(entity, "OverallWidth", None))
        height = _as_float(getattr(entity, "OverallHeight", None))
        if width is None:
            width = _as_float(_first_prop(props, ("OverallWidth", "Width", "宽度", "净宽")))
        if height is None:
            height = _as_float(_first_prop(props, ("OverallHeight", "Height", "高度")))
        width = round(width * length_to_m, 3) if width is not None else None
        height = round(height * length_to_m, 3) if height is not None else None
        fire_rating = _first_prop(props, ("FireRating", "FireResistanceRating", "FireResistance", "耐火"))
        operation = _first_prop(props, ("OperationType", "DoorOperationType", "开启方向", "Operation"))
        is_external = _first_prop(props, ("IsExternal", "External", "是否外门"))
        rows.append(
            {
                "global_id": _safe_str(getattr(entity, "GlobalId", "")),
                "name": _short(name) or "未命名门",
                "storey": _storey_name(entity, storey_index),
                "width_m": width,
                "height_m": height,
                "fire_rating": fire_rating,
                "operation": operation,
                "is_external": is_external,
                "is_exit_like": _has_any(name, EXIT_KEYWORDS),
                "raw_type": entity.is_a(),
            }
        )
    return rows


def _extract_windows(model: Any, storey_index: dict[str, str], length_to_m: float = 1.0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entity in _sample_entities(model, "IfcWindow", 200):
        props = _flatten_props(entity, max_items=60)
        name = _entity_name(entity)
        width = _as_float(getattr(entity, "OverallWidth", None))
        height = _as_float(getattr(entity, "OverallHeight", None))
        if width is None:
            width = _as_float(_first_prop(props, ("OverallWidth", "Width", "宽度", "净宽")))
        if height is None:
            height = _as_float(_first_prop(props, ("OverallHeight", "Height", "高度")))
        width = round(width * length_to_m, 3) if width is not None else None
        height = round(height * length_to_m, 3) if height is not None else None
        rows.append(
            {
                "global_id": _safe_str(getattr(entity, "GlobalId", "")),
                "name": _short(name) or "未命名窗",
                "storey": _storey_name(entity, storey_index),
                "width_m": width,
                "height_m": height,
                "raw_type": entity.is_a(),
            }
        )
    return rows


def _extract_elements_by_keywords(model: Any, storey_index: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    checked_types = [
        "IfcStair",
        "IfcRamp",
        "IfcTransportElement",
        "IfcFlowTerminal",
        "IfcFlowController",
        "IfcDistributionElement",
        "IfcSystem",
        "IfcWall",
        "IfcSlab",
    ]
    for ifc_type in checked_types:
        for entity in _sample_entities(model, ifc_type, 160):
            name = _entity_name(entity)
            if not _has_any(name, FIRE_KEYWORDS + STAIR_KEYWORDS + EXIT_KEYWORDS + GARAGE_KEYWORDS + ATRIUM_KEYWORDS):
                continue
            props = _flatten_props(entity, max_items=40)
            rows.append(
                {
                    "name": _short(name) or "未命名构件",
                    "ifc_type": entity.is_a(),
                    "storey": _storey_name(entity, storey_index),
                    "fire_rating": _first_prop(props, ("FireRating", "FireResistanceRating", "耐火")),
                }
            )
            if len(rows) >= 120:
                return rows
    return rows


def _building_height_from_storeys(storeys: list[dict[str, Any]]) -> float | None:
    elevations = [row["elevation_m"] for row in storeys if row.get("elevation_m") is not None]
    if len(elevations) < 2:
        return None
    return round(max(elevations) - min(elevations), 3)


def _storey_intervals(storeys: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        {"name": row.get("name") or "未命名楼层", "elevation_m": row.get("elevation_m")}
        for row in storeys
        if row.get("elevation_m") is not None
    ]
    rows = sorted(rows, key=lambda item: float(item["elevation_m"]))
    intervals: list[dict[str, Any]] = []
    for prev, current in zip(rows, rows[1:]):
        intervals.append(
            {
                "from": prev["name"],
                "to": current["name"],
                "height_m": round(float(current["elevation_m"]) - float(prev["elevation_m"]), 3),
            }
        )
    average = round(sum(item["height_m"] for item in intervals) / len(intervals), 3) if intervals else None
    return {"levels": rows, "intervals": intervals, "average_interval_m": average}


def _infer_building_use(path: str | Path, info: dict[str, Any], spaces: list[dict[str, Any]]) -> dict[str, str]:
    evidence_parts = [
        Path(path).name,
        info.get("project_name") or "",
        info.get("building_name") or "",
        info.get("site_name") or "",
    ]
    evidence_parts.extend([space.get("name") or "" for space in spaces[:80]])
    text = " ".join(evidence_parts).lower()
    if _has_any(text, GARAGE_KEYWORDS):
        return {"value": "疑似汽车库/停车相关建筑或局部空间", "confidence": "低", "reason": "命名或空间中出现停车相关词，但仍需人工确认建筑整体用途。"}
    if _has_any(text, RESIDENTIAL_KEYWORDS):
        return {"value": "疑似住宅/居住建筑", "confidence": "低", "reason": "文件名、项目名或空间名称出现居住相关词；IFC 未提供可直接作为审查依据的建筑使用功能。"}
    if _has_any(text, PUBLIC_KEYWORDS):
        return {"value": "疑似公共建筑", "confidence": "低", "reason": "命名或空间中出现公共建筑相关词；仍需人工确认具体使用功能。"}
    return {"value": "未稳定识别", "confidence": "低", "reason": "IFC 中未读取到可直接确定建筑使用功能的字段。"}


def _field_quality(summary: dict[str, Any]) -> dict[str, Any]:
    doors = summary.get("doors") or []
    spaces = summary.get("spaces") or []
    required = {
        "建筑使用功能": summary.get("building_use", {}).get("value") not in {"未稳定识别", None, ""},
        "楼层数量": bool(summary.get("counts", {}).get("IfcBuildingStorey")),
        "建筑高度": summary.get("building_height_m") is not None,
        "空间面积": any(space.get("area") is not None for space in spaces),
        "门宽": any(door.get("width_m") is not None for door in doors),
        "门用途": any(door.get("is_exit_like") for door in doors),
        "门防火性能": any(door.get("fire_rating") for door in doors),
    }
    available = [name for name, ok in required.items() if ok]
    missing = [name for name, ok in required.items() if not ok]
    return {
        "available": available,
        "missing": missing,
        "reviewability": "可做局部字段复核" if "门宽" in available or "建筑高度" in available else "只能做模型质量提示",
    }


def _unit_quality(summary: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    for door in summary.get("doors") or []:
        for field in ("width_m", "height_m"):
            value = _as_float(door.get(field))
            if value is not None and value <= 0:
                issues.append(f"{door.get('name') or '未命名门'} 的 {field} 为非正数，可能单位或建模异常")
    for space in summary.get("spaces") or []:
        area = _as_float(space.get("area"))
        if area is not None and (area <= 0 or area > 100000):
            issues.append(f"{space.get('name') or '未命名空间'} 的面积值异常，请人工确认面积单位")
    for storey in summary.get("storeys") or []:
        elevation = _as_float(storey.get("elevation_m"))
        if elevation is not None and abs(elevation) > 1000:
            issues.append(f"{storey.get('name') or '未命名楼层'} 的高程疑似非米单位")
    height = _as_float(summary.get("building_height_m"))
    if height is not None and height > 500:
        issues.append("建筑高度异常，疑似单位换算或建模错误，需人工确认")
    for interval in summary.get("storey_analysis", {}).get("intervals") or []:
        if _as_float(interval.get("height_m")) and _as_float(interval.get("height_m")) > 15:
            issues.append(f"{interval.get('from')} 至 {interval.get('to')} 层高异常，疑似单位换算或建模错误")
    project_units = summary.get("project_units") or {}
    units = {"长度": "m", "面积": "㎡", "高程": "m"}
    if project_units.get("length_label") and project_units["length_label"] != "m":
        units["IFC项目原始长度单位"] = project_units["length_label"]
    return {
        "reviewed": True,
        "units": units,
        "converted_from_project_unit": project_units.get("length_label", "m"),
        "issues": issues[:6],
    }


def _space_status(spaces: list[dict[str, Any]]) -> dict[str, Any]:
    if spaces:
        return {
            "status": "parsed",
            "count": len(spaces),
            "note": f"已解析 {len(spaces)} 个空间。",
        }
    return {
        "status": "not_defined",
        "count": 0,
        "note": "该模型未定义 IfcSpace；这是部分 IFC 的常见情况，不影响门、窗、楼梯等构件解析。",
    }


def parse_ifc_file(path: str | Path) -> dict[str, Any]:
    import ifcopenshell

    model = ifcopenshell.open(str(path))
    length_to_m, area_to_m2, length_label = _project_unit_factors(model)
    storey_index = _build_storey_index(model)
    info = _project_info(model)
    storeys = _extract_storeys(model, length_to_m)
    spaces = _extract_spaces(model, storey_index, area_to_m2)
    doors = _extract_doors(model, storey_index, length_to_m)
    windows = _extract_windows(model, storey_index, length_to_m)
    fire_elements = _extract_elements_by_keywords(model, storey_index)

    type_counts = {
        "IfcBuildingStorey": _count_type(model, "IfcBuildingStorey"),
        "IfcSpace": _count_type(model, "IfcSpace"),
        "IfcDoor": _count_type(model, "IfcDoor"),
        "IfcWindow": _count_type(model, "IfcWindow"),
        "IfcStair": _count_type(model, "IfcStair"),
        "IfcRamp": _count_type(model, "IfcRamp"),
        "IfcWall": _count_type(model, "IfcWall"),
        "IfcSlab": _count_type(model, "IfcSlab"),
        "IfcTransportElement": _count_type(model, "IfcTransportElement"),
    }
    space_counter = Counter()
    for space in spaces:
        if space.get("is_garage_related"):
            space_counter["garage"] += 1
        if space.get("is_atrium_related"):
            space_counter["atrium"] += 1

    summary = {
        "source_file": Path(path).name,
        "project": info,
        "project_units": {
            "length_label": length_label,
            "length_to_m": length_to_m,
            "area_to_m2": area_to_m2,
        },
        "counts": type_counts,
        "building_height_m": _building_height_from_storeys(storeys),
        "storey_analysis": _storey_intervals(storeys),
        "storeys": storeys,
        "spaces": spaces,
        "space_status": _space_status(spaces),
        "doors": doors,
        "windows": windows,
        "fire_related_elements": fire_elements,
        "detected_scenes": {
            "garage_spaces": space_counter["garage"],
            "atrium_spaces": space_counter["atrium"],
            "exit_like_doors": sum(1 for row in doors if row.get("is_exit_like")),
            "doors_with_width": sum(1 for row in doors if row.get("width_m") is not None),
            "doors_with_fire_rating": sum(1 for row in doors if row.get("fire_rating")),
        },
        "parser_limits": [
            "IFC 中很多消防语义依赖建模命名和属性集，未建模或命名不规范时无法自动判断。",
            "当前只抽取可稳定读取的几何尺寸、楼层、空间、门和消防相关构件字段，不做几何路径计算。",
            "疏散距离、房间到出口的真实路径、门开启方向等需要后续接入几何/拓扑计算后才能自动审查。",
        ],
    }
    summary["building_use"] = _infer_building_use(path, info, spaces)
    summary["field_quality"] = _field_quality(summary)
    summary["unit_quality"] = _unit_quality(summary)
    return summary


def build_ifc_review_questions(summary: dict[str, Any], limit: int = 6) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []
    counts = summary.get("counts") or {}
    scenes = summary.get("detected_scenes") or {}
    height = summary.get("building_height_m")
    storey_count = counts.get("IfcBuildingStorey", 0)

    if height is not None:
        building_use = summary.get("building_use", {})
        use_value = building_use.get("value") or "未稳定识别"
        questions.append(
            {
                "title": "建筑高度复核",
                "question": f"基于IFC解析：模型楼层高程推算建筑高度约 {height}m，共 {storey_count} 层；IFC名称推断的建筑用途为“{use_value}”。请结合当前用户补充的项目条件复核建筑分类及其影响的防火要求；仅对仍未明确的适用前提提示人工确认。",
            }
        )

    doors = [row for row in summary.get("doors", []) if row.get("width_m") is not None]
    seen_door_questions: set[tuple[str, float, str]] = set()
    unique_doors: list[dict[str, Any]] = []
    for door in doors:
        width = round(float(door.get("width_m")), 3)
        storey = door.get("storey") or "未识别楼层"
        role = "exit" if door.get("is_exit_like") else "unknown"
        key = (storey, width, role)
        if key in seen_door_questions:
            continue
        seen_door_questions.add(key)
        unique_doors.append(door)
        if len(unique_doors) >= 3:
            break

    if unique_doors:
        widths = [round(float(door.get("width_m")), 3) for door in unique_doors]
        minimum_door = min(unique_doors, key=lambda item: float(item.get("width_m")))
        minimum_name = _display_name(minimum_door.get("name") or "", fallback="未命名门")
        unknown_roles = sum(1 for door in doors if not door.get("is_exit_like"))
        questions.append(
            {
                "title": "门构件净宽复核",
                "question": f"基于IFC解析：模型中有 {len(doors)} 扇门可读取宽度，当前抽样宽度为 {widths}m，最小值约 {min(widths)}m（对象名：{minimum_name}）；其中 {unknown_roles} 扇不能仅凭IFC命名确认实际用途。请结合当前项目条件，先确定需要纳入疏散审查的门及其部位，再检索对应净宽要求并判断模型字段是否足以形成结论。",
            }
        )

    if scenes.get("garage_spaces"):
        questions.append(
            {
                "title": "汽车库专项复核",
                "question": f"基于IFC解析：模型中识别到 {scenes['garage_spaces']} 个疑似汽车库或停车相关空间。汽车库场景下应优先核对哪些安全疏散和消防设施要求？",
            }
        )

    if scenes.get("atrium_spaces"):
        questions.append(
            {
                "title": "中庭防火分隔复核",
                "question": f"基于IFC解析：模型中识别到 {scenes['atrium_spaces']} 个疑似中庭空间。中庭上下层连通时防火分区面积和防火分隔应如何复核？",
            }
        )

    if counts.get("IfcStair", 0):
        questions.append(
            {
                "title": "疏散楼梯复核",
                "question": f"基于IFC解析：模型中识别到 {counts['IfcStair']} 个楼梯构件。请说明疏散楼梯净宽、数量和封闭形式在消防规范中应如何复核。",
            }
        )

    if not questions:
        questions.append(
            {
                "title": "模型信息不足复核",
                "question": "基于IFC解析：当前模型未稳定提取到可直接用于数值审查的消防字段。请说明进行建筑消防合规检查前，IFC 模型至少应补充哪些字段？",
            }
        )

    return questions[:limit]


def ifc_processing_chain() -> list[dict[str, str]]:
    return [
        {
            "stage": "1. IFC 解析",
            "detail": "使用 IfcOpenShell 读取 IFC 文件，按 IfcBuildingStorey、IfcSpace、IfcDoor、IfcStair、IfcTransportElement 等类型提取对象。",
        },
        {
            "stage": "2. 字段归一",
            "detail": "把 OverallWidth、OverallHeight、Elevation、FireRating、IsExternal、空间名称、楼层归属等字段统一成可审查事实。",
        },
        {
            "stage": "3. 场景识别",
            "detail": "根据 IFC 类型、属性集和对象命名识别疑似汽车库、中庭、疏散出口、楼梯、消防设施等场景；这里只做事实标签，不给合规结论。",
        },
        {
            "stage": "4. 生成审查问题",
            "detail": "把结构化事实转成自然语言审查问题，例如“某门宽度 0.78m 是否满足疏散出口门最低要求”。",
        },
        {
            "stage": "5. RAG 检索规范",
            "detail": "审查问题进入现有 BM25 + FAISS + rerank 的规范知识库，召回标准号、条文号、页码和原文片段。",
        },
        {
            "stage": "6. 合规判断",
            "detail": "大语言模型只基于召回原文和 IFC 事实给结论；证据不足时必须提示人工复核。",
        },
    ]
