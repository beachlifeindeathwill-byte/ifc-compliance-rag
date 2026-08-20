from __future__ import annotations

import re


PHRASE_GROUPS = [
    {
        "triggers": ["商业综合体", "商业建筑", "商场", "商店营业厅", "展览厅"],
        "expansions": ["商业建筑", "公共建筑"],
    },
    {
        "triggers": ["中庭", "中庭四周", "共享空间"],
        "expansions": ["中庭", "周围连通空间", "相连通", "上下层相连通", "防火分隔", "防火分区"],
    },
    {
        "triggers": ["防火卷帘", "卷帘"],
        "expansions": ["防火卷帘", "防火分隔", "耐火极限", "自行关闭", "烟密闭"],
    },
    {
        "triggers": ["自动扶梯", "敞开楼梯", "敞开楼梯间", "开口", "上下连通"],
        "expansions": ["上下层相连通", "开口", "防火分区", "叠加计算", "防火分隔"],
    },
    {
        "triggers": ["汽车库", "地下汽车库", "修车库", "停车场"],
        "expansions": ["汽车库", "修车库"],
    },
    {
        "triggers": ["汽车库防火分类", "汽车库分类", "修车库分类", "停车场分类"],
        "expansions": ["停车数量", "车位数量", "总建筑面积"],
    },
    {
        "triggers": ["民用建筑分类", "高层类别", "一类高层", "二类高层"],
        "expansions": ["民用建筑", "建筑高度", "使用功能", "高层民用建筑", "一类", "二类"],
    },
    {
        "triggers": ["耐火等级", "燃烧性能", "耐火极限"],
        "expansions": ["耐火等级", "燃烧性能", "耐火极限", "构件"],
    },
    {
        "triggers": ["消防用水量", "用水量"],
        "expansions": ["室内消防用水量", "室外消防用水量", "消火栓", "自动喷水", "泡沫"],
    },
    {
        "triggers": ["排烟", "排烟系统", "排烟设施", "防烟分区"],
        "expansions": ["排烟设施", "排烟系统", "防烟分区", "划分防烟分区"],
    },
    {
        "triggers": ["安全出口", "人员安全出口", "疏散出口", "汽车疏散出口"],
        "expansions": ["人员安全出口", "汽车疏散出口", "疏散楼梯", "分开设置", "不应少于"],
    },
    {
        "triggers": ["每个防火分区", "每个楼层", "出口数量", "几个安全出口", "多少个安全出口"],
        "expansions": ["每个防火分区", "每个楼层", "安全出口不应少于", "疏散楼梯"],
    },
    {
        "triggers": ["疏散距离", "最近人员安全出口", "安全出口距离"],
        "expansions": ["室内任一点", "最近人员安全出口", "疏散距离"],
    },
    {
        "triggers": ["高层医疗建筑", "医疗建筑", "病房楼"],
        "expansions": ["高层医疗建筑", "病房楼", "疏散楼梯", "疏散走道", "首层疏散外门"],
    },
    {
        "triggers": ["住宅建筑", "多层住宅", "高层住宅", "居住建筑"],
        "expansions": ["住宅建筑"],
    },
    {
        "triggers": ["住宅户门", "入户门", "户门", "直通室外地面"],
        "expansions": ["住宅建筑", "住宅户门", "户门", "通行净宽", "净宽度"],
    },
    {
        "triggers": ["疏散门"],
        "expansions": ["疏散门", "疏散出口门", "房间疏散门"],
    },
    {
        "triggers": ["避难", "避难层", "避难间"],
        "expansions": ["避难层", "避难间", "消防设施", "疏散"],
    },
    {
        "triggers": ["基本目标", "制定目的", "制定的目标", "为什么制定"],
        "expansions": ["制定本规范", "预防建筑火灾", "减少火灾危害", "保障人身和财产安全"],
    },
]


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def expand_query_for_retrieval(query: str) -> str:
    """Add domain synonyms for retrieval only; do not add facts or answers."""
    compact = compact_text(query)
    additions: list[str] = []

    for group in PHRASE_GROUPS:
        if not any(trigger in compact for trigger in group["triggers"]):
            continue
        for term in group["expansions"]:
            if term not in compact and term not in additions:
                additions.append(term)

    if any(term in compact for term in ["汽车库", "修车库", "停车场"]) and any(term in compact for term in ["分类", "类别", "划分"]):
        for term in ["停车数量", "车位数量", "总建筑面积"]:
            if term not in compact and term not in additions:
                additions.append(term)

    if "民用建筑" in compact and any(term in compact for term in ["分类", "类别", "划分"]):
        for term in ["建筑高度", "使用功能", "高层民用建筑", "一类", "二类"]:
            if term not in compact and term not in additions:
                additions.append(term)

    if "消防电梯前室" in compact and any(term in compact for term in ["门", "防火性能", "耐火性能"]):
        for term in ["防火门", "乙级防火门", "耐火性能"]:
            if term not in compact and term not in additions:
                additions.append(term)

    if "消防电梯前室" in compact and any(term in compact for term in ["使用面积", "短边", "尺寸"]):
        for term in ["前室", "使用面积", "短边"]:
            if term not in compact and term not in additions:
                additions.append(term)

    if not additions:
        return query
    return query.rstrip() + " " + " ".join(additions)
