from __future__ import annotations

import json
import re
from pathlib import Path

from evaluate_retrieval import bm25_score, build_index, is_hit, load_jsonl, snippet, tokenize
from retrieval_policy import infer_route, is_abolished, load_policy, policy_bonus


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "processed_text" / "chunks.jsonl"
DEFAULT_EVAL_PATH = ROOT / "data" / "eval_sets" / "fire_code_eval_v2.jsonl"
FALLBACK_EVAL_PATH = ROOT / "data" / "eval_sets" / "fire_code_eval_v1.jsonl"
EVAL_PATH = DEFAULT_EVAL_PATH if DEFAULT_EVAL_PATH.exists() else FALLBACK_EVAL_PATH
OUT_DIR = ROOT / "data" / "retrieval_runs"

DOMAIN_TERMS = [
    "商业综合体",
    "商业建筑",
    "商店营业厅",
    "中庭",
    "周围连通空间",
    "上下层相连通",
    "防火分隔",
    "防火卷帘",
    "建筑高度大于100米",
    "建筑高度大于100m",
    "避难层",
    "室外消火栓",
    "室内消火栓",
    "自动灭火系统",
    "自动喷水灭火系统",
    "火灾自动报警系统",
    "通信接口",
    "通信协议",
    "防火墙",
    "防火隔墙",
    "楼地面基层",
    "楼板",
    "屋面板",
    "开口",
    "疏散照明",
    "消防设施与器材",
    "消防电梯前室",
    "合用前室",
    "入户门",
    "户门",
    "轿厢门",
    "通行净宽",
    "使用面积",
    "短边",
    "乙级防火门",
    "耐火性能",
    "柴油发电机房",
    "疏散出口门",
    "疏散出口",
    "防火分区",
    "汽车库",
    "修车库",
    "排烟系统",
    "排烟设施",
    "防烟分区",
    "泡沫灭火系统",
    "全淹没二氧化碳灭火系统",
    "灭火器",
    "消防车道",
    "消防车登高操作场地",
    "分类",
    "停车数量",
    "车位数量",
    "总建筑面积",
    "建筑分类",
    "耐火等级",
    "消防用水量",
    "室内消防用水量",
    "室外消防用水量",
    "叠加计算",
    "每个楼层",
    "每个防火分区",
    "位置",
    "宽度",
    "使用功能",
    "火灾危险性",
    "人员密度",
    "基本目标",
    "制定目的",
    "制定本规范",
]


def extract_query_terms(text: str) -> list[str]:
    candidates = [
        "商业综合体",
        "商业建筑",
        "商店营业厅",
        "中庭",
        "周围连通空间",
        "上下层相连通",
        "防火分隔",
        "防火卷帘",
        "建筑高度",
        "公共建筑",
        "民用建筑",
        "住宅建筑",
        "入户门",
        "户门",
        "轿厢门",
        "通行净宽",
        "医疗建筑",
        "地下汽车库",
        "汽车库",
        "防火分区",
        "最大允许建筑面积",
        "疏散出口门",
        "疏散出口",
        "疏散门",
        "疏散走道",
        "疏散楼梯",
        "安全出口",
        "相邻两个",
        "水平距离",
        "净宽度",
        "耐火极限",
        "楼板",
        "防火墙",
        "防火隔墙",
        "楼地面基层",
        "楼板",
        "屋面板",
        "开口",
        "门窗洞口",
        "自动灭火系统",
        "剪刀楼梯间",
        "消防电梯",
        "前室",
        "消防电梯前室",
        "合用前室",
        "使用面积",
        "短边",
        "防火性能",
        "防火门",
        "乙级防火门",
        "分类",
        "停车数量",
        "车位数量",
        "总建筑面积",
        "高层民用建筑",
        "一类",
        "二类",
        "耐火等级",
        "排烟系统",
        "排烟设施",
        "防烟分区",
        "消防用水量",
        "室内消防用水量",
        "室外消防用水量",
        "汽车疏散出口",
        "叠加计算",
        "每个楼层",
        "每个防火分区",
        "位置",
        "宽度",
        "使用功能",
        "火灾危险性",
        "人员密度",
        "基本目标",
        "制定目的",
        "制定本规范",
    ]
    return [term for term in candidates if term in text]


FOCUS_ATTRIBUTES = (
    "最大允许建筑面积",
    "通行净宽",
    "净宽度",
    "净宽",
    "使用面积",
    "耐火极限",
    "防火性能",
    "疏散距离",
    "水平距离",
    "建筑面积",
    "面积",
    "距离",
    "数量",
    "宽度",
)


def extract_focus_pair(question: str) -> tuple[str, str] | None:
    """Extract the object and requested attribute nearest to each other."""
    compact = compact_for_match(question)
    terms = sorted(set(extract_query_terms(compact)), key=len, reverse=True)
    best: tuple[int, int, int, str, str] | None = None
    for attribute in FOCUS_ATTRIBUTES:
        attribute_index = compact.find(attribute)
        if attribute_index < 0:
            continue
        for term in terms:
            if term == attribute or term in FOCUS_ATTRIBUTES or attribute in term:
                continue
            term_index = compact.rfind(term, 0, attribute_index)
            if term_index < 0:
                continue
            gap = attribute_index - (term_index + len(term))
            if gap < 0 or gap > 6:
                continue
            rank = (gap, -len(attribute), -len(term), term, attribute)
            if best is None or rank < best:
                best = rank
    if best is None:
        return None
    return best[3], best[4]


BUILDING_TYPE_TERMS = {
    "住宅": ["住宅", "居住", "户门", "套内", "卧室", "公寓"],
    "医疗": ["医疗", "医院", "病房", "门诊"],
    "汽车库": ["汽车库", "修车库", "停车场"],
    "商业": ["商业", "商场", "商店", "营业厅", "综合体"],
    "厂房": ["厂房", "车间"],
    "仓库": ["仓库", "库房"],
}


def detect_building_types(text: str) -> list[str]:
    compact = re.sub(r"\s+", "", text or "")
    return [key for key, terms in BUILDING_TYPE_TERMS.items() if any(term in compact for term in terms)]


def building_type_boost(question: str, chunk: dict) -> tuple[float, list[str]]:
    query_types = detect_building_types(question)
    if not query_types:
        return 0.0, []
    chunk_types = detect_building_types(chunk.get("text", "")[:400])
    if not chunk_types:
        return 0.0, []
    matched = [item for item in query_types if item in chunk_types]
    missing = [item for item in query_types if item not in chunk_types]
    notes: list[str] = []
    bonus = 0.0
    if matched:
        bonus += 10.0
        notes.append(f"building-type: 命中建筑类型 {'/'.join(matched)}")
    if missing:
        bonus -= 12.0
        notes.append(f"building-type: 未命中建筑类型 {'/'.join(missing)}")
    return bonus, notes


def compact_for_match(text: str) -> str:
    return re.sub(r"[\s、，,。；;：:（）()《》“”\"'`]+", "", text or "")


def leading_clause_text(compact: str, article_no: str) -> str:
    if article_no and compact.startswith(article_no):
        return compact[len(article_no) : len(article_no) + 260]
    return compact[:260]


def longest_common_substring_length(left: str, right: str, *, limit: int = 120) -> int:
    left = left[:limit]
    right = right[:limit]
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    best = 0
    for char_left in left:
        current = [0] * (len(right) + 1)
        for idx, char_right in enumerate(right, start=1):
            if char_left == char_right:
                current[idx] = previous[idx - 1] + 1
                best = max(best, current[idx])
        previous = current
    return best


def local_rerank_bonus(question: str, chunk: dict) -> tuple[float, list[str]]:
    text = chunk.get("text", "")
    notes: list[str] = []
    bonus = 0.0
    type_bonus, type_notes = building_type_boost(question, chunk)
    bonus += type_bonus
    notes.extend(type_notes)

    if chunk.get("article_no"):
        bonus += 0.8
        notes.append("rerank: 条文片段优先")
    else:
        penalty = 2.0 if chunk.get("has_table") else 4.0
        bonus -= penalty
        notes.append("rerank: 无条文号片段降权")

    compact_question = compact_for_match(question)
    compact = compact_for_match(text)
    article_no_raw = str(chunk.get("article_no") or "")
    first_article_part = article_no_raw.split(".", 1)[0]
    if first_article_part.isdigit() and int(first_article_part) > 20:
        bonus -= 16.0
        notes.append("rerank: 异常条文号降权")

    asks_width_value = any(term in compact_question for term in ["净宽", "宽度"]) and any(
        term in compact_question for term in ["多少", "最低", "最小", "分别", "复核"]
    )
    if asks_width_value:
        has_width_value = bool(re.search(r"\d+(?:\.\d+)?\s*(?:m|米)", text, flags=re.IGNORECASE))
        if has_width_value:
            bonus += 12.0
            notes.append("rerank: 净宽数值题优先含数值条款")
        else:
            bonus -= 4.0
            notes.append("rerank: 净宽数值题候选缺少数值")
    if compact.startswith(("前言", "目录", "目次")) or "深化工程建设标准化工作改革" in text:
        bonus -= 8.0
        notes.append("rerank: 前言/目录类片段降权")

    explanation_markers = ["本条为强制性条文", "本条规定", "本条说明", "条文说明", "有关说明参见"]
    if any(marker in text[:160] for marker in explanation_markers):
        bonus -= 3.0
        notes.append("rerank: 条文说明/解释性片段降权")
    article_no = compact_for_match(str(chunk.get("article_no") or ""))
    if article_no and (
        compact.startswith(article_no + "本条")
        or compact.startswith(article_no + "条")
        or compact.startswith(article_no + "有关说明")
    ):
        bonus -= 6.0
        notes.append("rerank: 条文说明续段降权")

    query_terms = extract_query_terms(compact_question)
    matched_terms = [term for term in query_terms if term in compact]
    if len(matched_terms) >= 2:
        bonus += min(18.0, len(matched_terms) * 3.5)
        notes.append(f"rerank: 命中多个问题字段 {'/'.join(matched_terms[:4])}")
    elif query_terms and not matched_terms:
        bonus -= 2.0
        notes.append("rerank: 未命中问题核心字段")

    lead = leading_clause_text(compact, article_no)
    lead_head = lead[:140]

    focus_pair = extract_focus_pair(compact_question)
    if focus_pair:
        focus_object, focus_attribute = focus_pair
        object_index = lead.find(focus_object)
        attribute_index = lead.find(focus_attribute)
        if object_index >= 0 and attribute_index >= 0:
            separation = abs(attribute_index - object_index)
            bonus += 26.0 if separation <= 48 else 14.0
            notes.append(f"rerank: 检查对象与待查属性对齐 {focus_object}/{focus_attribute}")

    lead_matches = [term for term in query_terms if len(term) >= 2 and term in lead_head]
    if lead_matches:
        bonus += min(20.0, len(lead_matches) * 5.0)
        notes.append(f"rerank: 条文开头命中问题主语 {'/'.join(lead_matches[:4])}")
    elif len(matched_terms) >= 3:
        bonus -= 4.0
        notes.append("rerank: 问题字段仅在正文后部出现")

    head_common_len = longest_common_substring_length(compact_question, lead_head, limit=90)
    if head_common_len >= 6:
        bonus += min(12.0, head_common_len * 0.8)
        notes.append("rerank: 条文开头与问题短语连续匹配")
    if head_common_len >= 10:
        bonus += min(20.0, (head_common_len - 8) * 2.0)
        notes.append("rerank: 条文开头保留问题中的完整限定条件")

    if any(term in compact_question for term in ["汽车库", "修车库", "停车场"]) and any(
        term in compact_question for term in ["分类", "划分", "类别"]
    ):
        matched_evidence = [term for term in ["分类", "停车数量", "车位数量", "总建筑面积"] if term in compact]
        if matched_evidence:
            bonus += min(14.0, len(matched_evidence) * 4.0)
            notes.append(f"rerank: 命中车库分类字段 {'/'.join(matched_evidence[:4])}")
        else:
            bonus -= 5.0
            notes.append("rerank: 未命中车库分类字段")

    if "民用建筑" in compact_question and any(term in compact_question for term in ["分类", "划分", "类别"]):
        matched_evidence = [term for term in ["民用建筑", "建筑高度", "使用功能", "高层民用建筑", "一类", "二类"] if term in compact]
        if matched_evidence:
            bonus += min(18.0, len(matched_evidence) * 4.0)
            notes.append(f"rerank: 命中民用建筑分类字段 {'/'.join(matched_evidence[:4])}")
        else:
            bonus -= 8.0
            notes.append("rerank: 未命中民用建筑分类字段")

    intent_groups = [
        (["耐火等级"], ["耐火等级", "燃烧性能", "耐火极限"]),
        (["排烟", "防烟分区"], ["排烟设施", "排烟系统", "防烟分区"]),
        (["消防用水量", "用水量"], ["消防用水量", "室内消防用水量", "室外消防用水量"]),
        (["中庭", "上下层相连通"], ["中庭", "上下层相连通", "叠加计算", "防火分区"]),
        (["防火隔墙", "隔断", "开口"], ["防火隔墙", "楼地面基层", "楼板", "屋面板", "开口", "防止火灾蔓延"]),
        (["消防电梯前室", "防火性能", "防火门"], ["消防电梯前室", "合用前室", "门的耐火性能", "乙级防火门"]),
        (["消防电梯前室", "使用面积", "短边"], ["消防电梯前室", "前室", "使用面积", "短边"]),
        (["灭火器", "配置类型"], ["灭火器", "配置类型", "火灾种类", "危险等级"]),
        (["数量", "位置", "宽度", "相适应"], ["数量", "位置", "宽度", "使用功能", "火灾危险性", "人员密度"]),
        (["每个防火分区", "每个楼层", "出口数量"], ["每个防火分区", "每个楼层", "安全出口不应少于", "疏散楼梯"]),
        (["基本目标", "制定"], ["预防建筑火灾", "减少火灾危害", "人身和财产安全", "制定本规范"]),
    ]
    for query_markers, evidence_markers in intent_groups:
        if not any(marker in compact_question for marker in query_markers):
            continue
        matched_evidence = [marker for marker in evidence_markers if marker in compact]
        if matched_evidence:
            bonus += min(12.0, len(matched_evidence) * 4.0)
            notes.append(f"rerank: 命中意图字段 {'/'.join(matched_evidence[:4])}")
        else:
            bonus -= 4.0
            notes.append("rerank: 未命中该问题意图字段")

    asks_for_applicable_rules = (
        "应符合哪些规定" in compact_question
        or "应满足哪些要求" in compact_question
        or "有什么要求" in compact_question
        or "如何确定" in compact_question
        or "如何设置" in compact_question
    )
    if asks_for_applicable_rules and article_no:
        leading_matches = [term for term in query_terms if term in lead]
        if len(leading_matches) >= 2:
            bonus += 8.0
            notes.append("rerank: 条文开头匹配问题主题")
        common_len = longest_common_substring_length(compact_question, lead)
        if common_len >= 12:
            bonus += min(14.0, common_len * 0.7)
            notes.append("rerank: 条文开头与问题长短语匹配")
        if "应符合下列规定" in lead or "应符合下列要求" in lead:
            bonus += 24.0
            notes.append("rerank: 条文标题匹配规定型问法")
        if "应分为" in lead and "应符合" not in lead and "分类" not in compact_question:
            bonus -= 12.0
            notes.append("rerank: 分类表片段与规定型问法不完全匹配")
        if "耐火等级" in compact_question and "防火间距" in lead:
            bonus -= 10.0
            notes.append("rerank: 耐火等级问题降低防火间距条文干扰")

    if "疏散出口门" in compact_question and "净宽度" in compact_question:
        if "疏散出口门" in lead and "净宽度" in lead and "不应小于" in compact:
            bonus += 24.0
            notes.append("rerank: 命中疏散出口门净宽度下限")
        elif "疏散出口门" not in lead:
            bonus -= 12.0
            notes.append("rerank: 疏散出口门净宽问题降低泛化疏散条文")

    if "消防电梯前室" in compact_question and "门" in compact_question:
        if "消防电梯前室" in compact and ("门的耐火性能" in compact or "乙级防火门" in compact):
            bonus += 24.0
            notes.append("rerank: 命中消防电梯前室门防火性能")
        elif "消防电梯前室" not in compact:
            bonus -= 18.0
            notes.append("rerank: 消防电梯前室门问题降低对象不完整候选")

    if "消防电梯前室" in compact_question and ("面积" in compact_question or "使用面积" in compact_question or "短边" in compact_question):
        if "消防电梯" in compact and "前室" in compact and ("使用面积" in compact or "短边" in compact):
            bonus += 24.0
            notes.append("rerank: 命中消防电梯前室面积尺寸")
        if "门的耐火性能" in compact or "乙级防火门" in compact:
            bonus -= 16.0
            notes.append("rerank: 前室面积问题降低门防火性能条文")

    if "防火隔墙" in compact_question:
        if lead.startswith("防火隔墙应") or ("防火隔墙应" in lead and "开口" in compact):
            bonus += 22.0
            notes.append("rerank: 命中防火隔墙本体要求")
        elif "防火隔墙" not in lead:
            bonus -= 10.0
            notes.append("rerank: 防火隔墙问题降低非本体条文")

    if "防火墙" in compact_question and "防火隔墙" not in compact_question:
        if lead.startswith("防火墙应") or ("防火墙应" in lead and "防止火灾蔓延" in compact):
            bonus += 18.0
            notes.append("rerank: 命中防火墙本体要求")

    if "消防车道" in compact_question:
        if "消防车道" in compact:
            bonus += 14.0
            notes.append("rerank: 命中消防车道对象")
            if "设置消防车道" in compact or "应设置环形消防车道" in compact or "车道的净宽度和净空高度" in compact:
                bonus += 12.0
                notes.append("rerank: 命中消防车道设置/通行要求")
        else:
            bonus -= 18.0
            notes.append("rerank: 消防车道问题降低无消防车道候选")
        if not any(term in compact_question for term in ["汽车库", "修车库", "停车场"]) and any(
            term in lead for term in ["汽车库", "修车库", "停车场"]
        ):
            bonus -= 10.0
            notes.append("rerank: 未指定车库时降低车库专项消防车道")

    if "消防车登高操作场地" in compact_question:
        if "消防车登高操作场地" in compact:
            bonus += 12.0
            notes.append("rerank: 命中消防车登高操作场地对象")
        else:
            bonus -= 12.0
            notes.append("rerank: 登高场地问题降低对象不完整候选")

    if "民用建筑之间" in compact_question and "防火间距" in compact_question:
        if "民用建筑之间的防火间距" in compact and "表" in compact:
            bonus += 22.0
            notes.append("rerank: 命中民用建筑之间防火间距表")
        if "木结构" in compact and "木结构" not in compact_question:
            bonus -= 14.0
            notes.append("rerank: 未指定木结构时降低木结构专项表")

    if "相适应" in compact_question and "疏散出口" in compact_question:
        if "相适应" in compact and "使用功能" in compact and "火灾危险性" in compact:
            bonus += 24.0
            notes.append("rerank: 命中疏散出口相适应因素")

    if "公共建筑" in compact_question and ("每个防火分区" in compact_question or "每个楼层" in compact_question):
        if "总净宽度" in compact or "每100人" in compact:
            bonus -= 16.0
            notes.append("rerank: 公共建筑出口数量问题降低净宽度计算条文")
        if "住宅与非住宅" in compact:
            bonus -= 12.0
            notes.append("rerank: 公共建筑出口数量问题降低住宅合建条文")

    if (
        "可设置1个" in compact_question
        or "设置1个" in compact_question
        or "只设1个" in compact_question
        or "只设一个" in compact_question
        or "设置一个" in compact_question
    ):
        if "可设置1个" in compact or "可设置1个" in lead:
            bonus += 16.0
            notes.append("rerank: 命中可设置1个条件")
        elif "分开设置" in lead and "可设置" not in compact:
            bonus -= 10.0
            notes.append("rerank: 设置1个问题降低分开设置原则")

    if "应设置自动喷水灭火系统" in compact_question:
        if "下列" in lead and "应设置自动喷水灭火系统" in compact:
            bonus += 18.0
            notes.append("rerank: 命中自动喷水设置范围")
        elif "设计除应符合" in compact or "喷头布置" in compact:
            bonus -= 10.0
            notes.append("rerank: 自动喷水设置范围问题降低设计细则")

    if "汽车库" in compact_question and "自动喷水灭火系统" in compact_question:
        if "应设置自动喷水灭火系统" in compact and "喷头布置" not in compact:
            bonus += 16.0
            notes.append("rerank: 命中汽车库自动喷水设置范围")
        if "喷头布置" in compact or "设计除应符合" in compact:
            bonus -= 14.0
            notes.append("rerank: 汽车库自动喷水范围问题降低喷头设计细则")

    if "灭火器" in compact_question and "配置类型" in compact_question:
        if "配置类型" in compact and "火灾种类" in compact and "危险等级" in compact:
            bonus += 22.0
            notes.append("rerank: 命中灭火器配置类型")
        elif "配置类型" not in compact:
            bonus -= 10.0
            notes.append("rerank: 灭火器配置类型问题降低非类型条文")

    if "排烟" in compact_question or "防烟分区" in compact_question:
        if "排烟" in compact and "防烟分区" in compact:
            bonus += 18.0
            notes.append("rerank: 排烟问题命中排烟和防烟分区")
        elif "排烟" not in compact:
            bonus -= 18.0
            notes.append("rerank: 排烟问题降低无排烟候选")
        if "分类" in lead and "排烟" not in lead:
            bonus -= 12.0
            notes.append("rerank: 排烟问题降低分类表干扰")

    asks_public_exit_count = "公共建筑" in compact_question and (
        "每个防火分区" in compact_question
        or "每个楼层" in compact_question
        or "出口数量" in compact_question
    )
    if asks_public_exit_count:
        if "公共建筑" in compact and (
            "每个防火分区" in compact or "每个楼层" in compact or "一个防火分区" in compact
        ):
            bonus += 36.0
            notes.append("rerank: 命中公共建筑分区/楼层出口数量场景")
            if "安全出口不应少于2个" in compact or "安全出口不应少于" in compact:
                bonus += 18.0
                notes.append("rerank: 命中公共建筑安全出口数量下限")
        elif "公共建筑" not in compact:
            bonus -= 14.0
            notes.append("rerank: 公共建筑出口数量问题降低泛化条文")

    if "\u51c0\u5bbd\u5ea6" in compact_question and (
        "\u6700\u4f4e" in compact_question or "\u6700\u5c0f" in compact_question or "\u4e0d\u5e94\u5c0f\u4e8e" in compact_question
    ):
        if "\u4e0d\u5e94\u5c0f\u4e8e" in compact and re.search(r"\d+(?:\.\d+)?\s*m", text, re.IGNORECASE):
            bonus += 4.0
            notes.append("rerank: 命中净宽度下限数值")
        if "\u51c0\u5bbd\u5ea6\u5e94\u7b26\u5408\u4e0b\u5217\u89c4\u5b9a" in compact:
            bonus += 3.0
            notes.append("rerank: 命中净宽度规定条款")

    asks_numeric_limit = any(term in compact_question for term in ["多少", "多大", "不应小于", "不应大于", "不宜大于", "不宜小于"])
    asks_numeric_judgement = bool(re.search(r"\d+(?:\.\d+)?", compact_question)) and any(
        term in compact_question for term in ["可以吗", "能不能", "是否可以", "是否合规", "能否", "接受"]
    )
    has_numeric_answer = bool(re.search(r"\d+(?:\.\d+)?\s*(?:m|h|㎡|m2|m²|个|人|辆)", text, re.IGNORECASE))
    if asks_numeric_limit or asks_numeric_judgement:
        numeric_markers = [
            term
            for term in [
                "水平距离",
                "相邻两个",
                "安全出口",
                "人员安全出口",
                "疏散出口门",
                "疏散门",
                "净宽",
                "净宽度",
                "使用面积",
                "短边",
                "疏散距离",
                "自动灭火系统",
            ]
            if term in compact_question
        ]
        numeric_hits = [term for term in numeric_markers if term in compact]
        if has_numeric_answer and len(numeric_hits) >= 2:
            bonus += min(18.0, 6.0 * len(numeric_hits))
            notes.append(f"rerank: 数值型问题命中对象字段 {'/'.join(numeric_hits[:4])}")
        elif numeric_markers and not has_numeric_answer:
            bonus -= 8.0
            notes.append("rerank: 数值型问题降低无明确数值候选")

    for term in DOMAIN_TERMS:
        if term in question and term in text:
            bonus += 3.0
            notes.append(f"rerank: 命中核心短语 {term}")

    return bonus, notes


def score_chunk(question: str, chunk: dict, query_tokens: list[str], doc_tokens: list[str], df: dict[str, int], total_docs: int, avgdl: float, route, registry: dict, abolished: dict) -> tuple[float, list[str]]:
    score = bm25_score(query_tokens, doc_tokens, df, total_docs, avgdl)
    notes: list[str] = []

    if chunk.get("has_table"):
        score += 0.15
    if chunk.get("article_no"):
        score += 0.2

    bonus, policy_notes = policy_bonus(question, chunk, route, registry, abolished)
    score += bonus
    notes.extend(policy_notes)
    rerank_bonus, rerank_notes = local_rerank_bonus(question, chunk)
    score += rerank_bonus
    notes.extend(rerank_notes)
    return score, notes


def main() -> None:
    registry, abolished = load_policy()
    chunks = load_jsonl(CHUNKS_PATH)
    questions = load_jsonl(EVAL_PATH)
    tokenized_docs, df, avgdl = build_index(chunks)
    total_docs = len(chunks)

    results = []
    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    mrr_total = 0.0
    deprecated_top1 = 0
    standard_at_1 = 0
    standard_at_3 = 0
    standard_at_5 = 0

    for question in questions:
        question_text = question["question"]
        route = infer_route(question_text, registry)
        query_tokens = tokenize(question_text)
        scored = []

        for chunk, doc_tokens in zip(chunks, tokenized_docs):
            score, policy_notes = score_chunk(
                question_text, chunk, query_tokens, doc_tokens, df, total_docs, avgdl, route, registry, abolished
            )
            if score > 0:
                scored.append((score, chunk, policy_notes))

        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[:10]

        first_hit_rank = None
        first_standard_rank = None
        for idx, (_, chunk, _) in enumerate(top, start=1):
            if is_hit(question, chunk):
                first_hit_rank = idx
                break
        for idx, (_, chunk, _) in enumerate(top, start=1):
            if question.get("expected_standard_id") and chunk["standard_id"] == question["expected_standard_id"]:
                first_standard_rank = idx
                break

        if first_hit_rank == 1:
            hit_at_1 += 1
        if first_hit_rank and first_hit_rank <= 3:
            hit_at_3 += 1
        if first_hit_rank and first_hit_rank <= 5:
            hit_at_5 += 1
        if first_hit_rank:
            mrr_total += 1 / first_hit_rank
        if first_standard_rank == 1:
            standard_at_1 += 1
        if first_standard_rank and first_standard_rank <= 3:
            standard_at_3 += 1
        if first_standard_rank and first_standard_rank <= 5:
            standard_at_5 += 1

        top1_deprecated = bool(top and is_abolished(top[0][1], abolished))
        if top1_deprecated:
            deprecated_top1 += 1

        results.append(
            {
                "id": question["id"],
                "question": question_text,
                "expected_standard_id": question.get("expected_standard_id"),
                "expected_article": question.get("expected_article"),
                "route": {
                    "primary": route.primary,
                    "secondary": route.secondary,
                    "reasons": route.reasons,
                },
                "first_hit_rank": first_hit_rank,
                "first_standard_rank": first_standard_rank,
                "top1_is_abolished_seed": top1_deprecated,
                "top_results": [
                    {
                        "rank": rank,
                        "score": round(score, 3),
                        "chunk_id": chunk["chunk_id"],
                        "standard_id": chunk["standard_id"],
                        "page": chunk["page"],
                        "article_no": chunk.get("article_no"),
                        "has_table": chunk.get("has_table"),
                        "is_hit_original_gold": is_hit(question, chunk),
                        "abolished_seed": is_abolished(chunk, abolished),
                        "policy_notes": policy_notes,
                        "snippet": snippet(chunk["text"]),
                    }
                    for rank, (score, chunk, policy_notes) in enumerate(top[:5], start=1)
                ],
            }
        )

    total = len(questions)
    summary = {
        "questions": total,
        "hit_at_1_original_gold": round(hit_at_1 / total, 3),
        "hit_at_3_original_gold": round(hit_at_3 / total, 3),
        "hit_at_5_original_gold": round(hit_at_5 / total, 3),
        "mrr_at_10_original_gold": round(mrr_total / total, 3),
        "standard_at_1": round(standard_at_1 / total, 3),
        "standard_at_3": round(standard_at_3 / total, 3),
        "standard_at_5": round(standard_at_5 / total, 3),
        "deprecated_top1_seed_count": deprecated_top1,
        "deprecated_top1_seed_rate": round(deprecated_top1 / total, 3),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "retrieval_eval_v2_policy.json").write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Retrieval Evaluation v2 Policy-Aware",
        "",
        f"- Eval set: `{EVAL_PATH.name}`",
        f"- Questions: {summary['questions']}",
        f"- Hit@1 original gold: {summary['hit_at_1_original_gold']}",
        f"- Hit@3 original gold: {summary['hit_at_3_original_gold']}",
        f"- Hit@5 original gold: {summary['hit_at_5_original_gold']}",
        f"- MRR@10 original gold: {summary['mrr_at_10_original_gold']}",
        f"- Standard@1: {summary['standard_at_1']}",
        f"- Standard@3: {summary['standard_at_3']}",
        f"- Standard@5: {summary['standard_at_5']}",
        f"- Top-1 deprecated seed count: {summary['deprecated_top1_seed_count']}",
        f"- Top-1 deprecated seed rate: {summary['deprecated_top1_seed_rate']}",
        "",
        "说明：original gold 仍使用第一版评估集标注；其中部分旧规范条文已被 seed 废止表标记，因此该指标不能单独代表最终法规正确性。",
        "",
        "## Routes And Weak Cases",
        "",
    ]

    for result in results:
        rank = result["first_hit_rank"]
        if rank is None or rank > 3 or result["top1_is_abolished_seed"]:
            lines.append(f"### {result['id']} {result['question']}")
            lines.append(f"Route primary: {', '.join(result['route']['primary']) or '全库'}")
            lines.append(f"Route secondary: {', '.join(result['route']['secondary']) or '无'}")
            lines.append(f"Route reasons: {'；'.join(result['route']['reasons'])}")
            lines.append(f"Expected original gold: {result['expected_standard_id']} / {result['expected_article']}")
            lines.append(f"First hit rank original gold: {rank}")
            lines.append(f"First standard rank: {result['first_standard_rank']}")
            for item in result["top_results"]:
                abolished_text = ""
                if item["abolished_seed"]:
                    abolished_text = f" abolished_seed -> {item['abolished_seed']['replaced_by_standard_id']}"
                lines.append(
                    f"- #{item['rank']} {item['standard_id']} p{item['page']} article={item['article_no']} "
                    f"hit={item['is_hit_original_gold']} score={item['score']}{abolished_text} `{item['chunk_id']}`"
                )
                notes = "；".join(item["policy_notes"])
                if notes:
                    lines.append(f"  Policy: {notes}")
                snippet = re.sub(r"\s+", " ", item["snippet"]).strip()
                lines.append(f"  {snippet}")
            lines.append("")

    (OUT_DIR / "retrieval_eval_v2_policy.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {OUT_DIR / 'retrieval_eval_v2_policy.md'}")


if __name__ == "__main__":
    main()
