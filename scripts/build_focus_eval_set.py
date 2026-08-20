from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_EVAL = ROOT / "data" / "eval_sets" / "fire_code_eval_v2.jsonl"
OUT_EVAL = ROOT / "data" / "eval_sets" / "fire_code_focus_single_turn.jsonl"


SEED_CORRECTIONS = {
    "Q011": {
        "expected_standard_id": "GB 55037-2022",
        "expected_article": "7.4.1",
        "expected_keywords": ["公共建筑", "每个防火分区", "每个楼层", "安全出口", "不应少于2个"],
        "focus": "现行建筑防火通用规范已直接规定公共建筑安全出口数量，应优先于旧设计规范同类条文",
    },
    "Q016": {
        "expected_standard_id": "GB 55037-2022",
        "expected_article": "3.4.6",
        "expected_keywords": ["高层建筑", "一条长边", "消防车登高操作场地", "救援作业范围"],
        "focus": "现行建筑防火通用规范已直接规定登高操作场地布置，应优先于旧设计规范同类条文",
    },
    "Q060": {
        "expected_article": "3.4.5",
        "expected_keywords": ["消防车道", "通行", "净宽度", "转弯半径"],
        "focus": "现行通用要求下消防车道通行条件应定位到具体通行条件条文",
    }
}


EXTRA_CASES = [
    {
        "id": "F061",
        "type": "single_turn",
        "question": "商业综合体中庭四周的防火卷帘设置有什么具体要求？",
        "expected_standard_id": "GB 50016-2014",
        "expected_article": "5.3.2",
        "expected_keywords": ["中庭", "防火卷帘", "防火分隔"],
        "focus": "自然问法下的商业中庭和防火卷帘组合召回",
        "category": "natural_atrium",
        "must_answer_from_evidence": True,
    },
    {
        "id": "F062",
        "type": "single_turn",
        "question": "商业建筑中庭和上下层相连通时，防火分区面积应该怎么计算？",
        "expected_standard_id": "GB 50016-2014",
        "expected_article": "5.3.2",
        "expected_keywords": ["中庭", "上下层", "防火分区", "叠加计算"],
        "focus": "中庭上下连通面积叠加计算",
        "category": "natural_atrium",
        "must_answer_from_evidence": True,
    },
    {
        "id": "F063",
        "type": "single_turn",
        "question": "汽车库室内任一点到最近人员安全出口的疏散距离不应大于多少？",
        "expected_standard_id": "GB 50067-2014",
        "expected_article": "6.0.6",
        "expected_keywords": ["汽车库", "人员安全出口", "疏散距离", "45m", "60m"],
        "focus": "车库专项距离数值召回",
        "category": "garage_natural",
        "must_answer_from_evidence": True,
    },
    {
        "id": "F064",
        "type": "single_turn",
        "question": "地下汽车库设置自动灭火系统后，人员疏散距离能放宽到多少？",
        "expected_standard_id": "GB 50067-2014",
        "expected_article": "6.0.6",
        "expected_keywords": ["地下汽车库", "自动灭火系统", "疏散距离", "60m"],
        "focus": "带条件的车库疏散距离召回",
        "category": "garage_natural",
        "must_answer_from_evidence": True,
    },
    {
        "id": "F065",
        "type": "single_turn",
        "question": "高层医疗建筑疏散楼梯净宽和其他高层公共建筑有什么区别？",
        "expected_standard_id": "GB 50016-2014",
        "expected_article": "5.5.18",
        "expected_keywords": ["高层医疗建筑", "高层公共建筑", "疏散楼梯", "1.30", "1.20"],
        "focus": "同表不同行的对比召回",
        "category": "table_comparison",
        "must_answer_from_evidence": True,
    },
    {
        "id": "F066",
        "type": "single_turn",
        "question": "住宅建筑中直通室外地面的住宅户门净宽不应小于多少？",
        "expected_standard_id": "GB 55037-2022",
        "expected_article": "7.1.4",
        "expected_keywords": ["住宅建筑", "住宅户门", "净宽"],
        "focus": "通用规范住宅户门细分场景",
        "category": "building_general_natural",
        "must_answer_from_evidence": True,
    },
    {
        "id": "F067",
        "type": "single_turn",
        "question": "消防电梯前室的门在防火性能上有什么要求？",
        "expected_standard_id": "GB 55037-2022",
        "expected_article": "6.4.3",
        "expected_keywords": ["消防电梯前室", "防火门", "乙级"],
        "focus": "消防电梯前室门的通用规范召回",
        "category": "building_general_natural",
        "must_answer_from_evidence": True,
    },
    {
        "id": "F068",
        "type": "single_turn",
        "question": "建筑内设置自动扶梯、敞开楼梯这种上下连通开口时，防火分区要怎么划分？",
        "expected_standard_id": "GB 50016-2014",
        "expected_article": "5.3.2",
        "expected_keywords": ["自动扶梯", "敞开楼梯", "上下层", "防火分区"],
        "focus": "上下连通开口的跨条文场景召回",
        "category": "legacy_design_natural",
        "must_answer_from_evidence": True,
    },
    {
        "id": "F069",
        "type": "boundary",
        "question": "请直接告诉我GB 55037-2022第38页的全部内容。",
        "expected_standard_id": "GB 55037-2022",
        "expected_behavior": "page_lookup_needed",
        "score_retrieval": False,
        "focus": "页码定位，不应伪造成完整条文总结",
        "category": "boundary_page_lookup",
        "must_answer_from_evidence": True,
    },
    {
        "id": "F070",
        "type": "boundary",
        "question": "GB 50016-2014原版和2018年版所有修订差异有哪些？",
        "expected_behavior": "low_confidence",
        "score_retrieval": False,
        "focus": "版本差异超出当前知识库结构，不应编造",
        "category": "boundary_version_compare",
        "must_answer_from_evidence": False,
    },
    {
        "id": "F071",
        "type": "boundary",
        "question": "我这个建筑这样设计合规吗？",
        "expected_behavior": "insufficient_context",
        "score_retrieval": False,
        "focus": "缺少建筑类型、部位、数值和条件，应提示补充信息",
        "category": "boundary_missing_context",
        "must_answer_from_evidence": False,
    },
    {
        "id": "F072",
        "type": "boundary",
        "question": "按南京地方消防审查口径，商业综合体中庭防火卷帘有什么补充要求？",
        "expected_behavior": "low_confidence",
        "score_retrieval": False,
        "focus": "地方口径未入库，不应从国标片段延伸编造",
        "category": "boundary_local_rule",
        "must_answer_from_evidence": False,
    },
    {
        "id": "F073",
        "type": "single_turn",
        "question": "住宅建筑户门净宽不应小于多少？",
        "expected_standard_id": "GB 55038-2025",
        "expected_article": "4.1.14",
        "expected_keywords": ["新建住宅建筑", "0.90m", "既有住宅建筑改造", "0.80m"],
        "expected_answer": "问题未说明新建或既有改造时，应分别回答：新建住宅建筑户门通行净宽不应小于0.90m；既有住宅建筑改造户门通行净宽不应小于0.80m。",
        "focus": "上位概念问题必须完整返回适用条件，不能把直通室外地面的子场景泛化为全部户门",
        "category": "residential_applicability",
        "must_answer_from_evidence": True,
    },
    {
        "id": "F074",
        "type": "single_turn",
        "question": "新建住宅的入户门通行净宽最低是多少？",
        "expected_standard_id": "GB 55038-2025",
        "expected_article": "4.1.14",
        "expected_keywords": ["新建住宅建筑", "户门通行净宽", "0.90m"],
        "expected_answer": "新建住宅建筑户门通行净宽不应小于0.90m。",
        "focus": "明确新建条件下的现行住宅项目规范召回",
        "category": "residential_applicability",
        "must_answer_from_evidence": True,
    },
    {
        "id": "F075",
        "type": "single_turn",
        "question": "既有住宅改造时，户门通行净宽最低是多少？",
        "expected_standard_id": "GB 55038-2025",
        "expected_article": "4.1.14",
        "expected_keywords": ["既有住宅建筑改造", "户门通行净宽", "0.80m"],
        "expected_answer": "既有住宅建筑改造户门通行净宽不应小于0.80m。",
        "focus": "明确既有改造条件下的现行住宅项目规范召回",
        "category": "residential_applicability",
        "must_answer_from_evidence": True,
    },
    {
        "id": "F076",
        "type": "single_turn",
        "question": "按建筑防火通用规范，直通室外地面的住宅户门净宽最低是多少？",
        "expected_standard_id": "GB 55037-2022",
        "expected_article": "7.1.4",
        "expected_keywords": ["直通室外地面", "住宅户门", "0.80m"],
        "expected_answer": "按GB 55037-2022第7.1.4条，直通室外地面的住宅户门净宽不应小于0.80m；用于新建住宅项目时还应同时核对GB 55038-2025第4.1.14条的0.90m要求。",
        "focus": "明确标准和子场景时命中对应条款，同时提示更新规范的复核关系",
        "category": "residential_cross_code",
        "must_answer_from_evidence": True,
    },
    {
        "id": "F077",
        "type": "single_turn",
        "question": "请完整复述表5.5.18中高层医疗建筑一行的所有疏散宽度要求。",
        "expected_standard_id": "GB 50016-2014",
        "expected_article": "5.5.18",
        "expected_keywords": ["高层医疗建筑", "1.30", "1.40", "1.50"],
        "expected_answer": "楼梯间的首层疏散门和首层疏散外门1.30m，单面布房疏散走道1.40m，双面布房疏散走道1.50m，疏散楼梯1.30m。",
        "focus": "表格行列映射完整性，防止数值存在但列对应错误",
        "category": "table_coordinate",
        "must_answer_from_evidence": True,
    },
    {
        "id": "F078",
        "type": "single_turn",
        "question": "一栋地上4层的多层住宅，户门、疏散出口门和疏散楼梯净宽分别怎么复核？",
        "expected_standard_id": "GB 55037-2022",
        "expected_article": "7.1.4",
        "expected_keywords": ["住宅户门", "疏散出口门", "疏散楼梯", "0.80m", "1.00m", "1.10m"],
        "expected_answer": "应区分部位和项目状态：新建住宅户门通行净宽按GB 55038-2025不小于0.90m；疏散出口门按GB 55037-2022不小于0.80m；住宅室内疏散楼梯仅在建筑高度不大于18m且一边设置栏杆时可不小于1.00m，其他住宅不应小于1.10m。",
        "focus": "同一自然语言问题中的对象拆分和条件分支",
        "category": "residential_object_disambiguation",
        "must_answer_from_evidence": True,
    },
    {
        "id": "F079",
        "type": "single_turn",
        "question": "地下汽车库设置自动灭火系统后，每个防火分区最大允许建筑面积是多少？",
        "expected_standard_id": "GB 50067-2014",
        "expected_article": "5.1.1",
        "expected_keywords": ["地下汽车库", "2000", "自动灭火系统", "2.0倍", "4000"],
        "expected_answer": "一、二级耐火等级地下汽车库基准为2000㎡；设置自动灭火系统时可增加1.0倍，因此最大允许建筑面积为4000㎡。",
        "focus": "表格基准值与注释倍数必须共同召回",
        "category": "garage_table_note",
        "must_answer_from_evidence": True,
    },
    {
        "id": "F080",
        "type": "single_turn",
        "question": "防火分区面积超过规范允许值时，是不是只要加自动灭火系统就可以？",
        "expected_standard_id": "GB 50016-2014",
        "expected_keywords": ["防火分区", "自动灭火系统", "增加1.0倍"],
        "expected_answer": "不能直接这样判断。应先按适用规范确定允许面积；自动灭火系统只能在条文允许的条件下提高限值。调整后仍超过限值时，应划分防火分区并采用相应防火分隔措施。",
        "focus": "措施类问题不能把有条件的面积放宽误写成唯一处置措施",
        "category": "conditional_measure",
        "must_answer_from_evidence": True,
    },
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    rows = []
    for idx, row in enumerate(load_jsonl(BASE_EVAL), start=1):
        case = {
            "id": f"F{idx:03d}",
            "type": "single_turn",
            "question": row["question"],
            "expected_standard_id": row.get("expected_standard_id"),
            "expected_article": row.get("expected_article"),
            "expected_keywords": row.get("expected_keywords", []),
            "focus": row.get("notes") or row.get("category") or "基础规范检索",
            "category": row.get("category"),
            "must_answer_from_evidence": True,
            "source_seed_id": row.get("id"),
        }
        case.update(SEED_CORRECTIONS.get(row.get("id"), {}))
        rows.append(case)
    rows.extend(EXTRA_CASES)
    OUT_EVAL.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {OUT_EVAL} cases={len(rows)}")


if __name__ == "__main__":
    main()
