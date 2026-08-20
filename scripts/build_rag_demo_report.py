from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "data" / "retrieval_runs"
OUT_PATH = ROOT / "docs" / "rag_evaluation_report.md"


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_rate(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    focus = read_json(RUN_DIR / "focus_single_turn_eval.json")
    natural = read_json(RUN_DIR / "natural_scenarios_eval.json")
    multiturn = read_json(RUN_DIR / "multiturn_eval.json")
    chunk_summary = read_json(ROOT / "data" / "processed_text" / "chunk_summary.json")
    vector_config = read_json(ROOT / "data" / "vectorstore" / "embedding_config.json")

    focus_summary = focus.get("summary", {})
    natural_summary = natural.get("summary", {})
    multiturn_summary = multiturn.get("summary", {})
    chunks = chunk_summary.get("summaries", [])

    lines = [
        "# 建筑消防规范 RAG 评估报告",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 项目目标",
        "",
        "本项目面向建筑消防规范问答与后续 IFC 审查场景。当前阶段先验证“规范文本能否被稳定召回、回答是否可追溯、证据不足时是否拒答”，再进入 IFC 模型解析。",
        "",
        "## 2. 当前知识库范围",
        "",
        "| 文件 | 标准号 | 页数 | chunk 数 | 含条文号 chunk | 表格 chunk | 切片策略 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in chunks:
        lines.append(
            "| {file} | {standard_id} | {pages} | {chunks} | {chunks_with_article_no} | {chunks_with_table} | max {max_chunk_chars}, overlap {overlap_chars} |".format(
                **item
            )
        )

    lines.extend(
        [
            "",
            "当前页码口径：系统展示的是 **PDF 物理页**，包含封面、目录等前置页面；不是规范正文自带页码。",
            "",
            "## 3. RAG 流程",
            "",
            "1. PDF 文本抽取与清洗：保留标准号、来源文件、PDF 页码、条文号、表号等 metadata。",
            "2. 条文优先切片：优先按条文边界切分，不只按固定字数切；跨页续条文会继承父条文号。",
            "3. 表格独立索引：表格内容单独形成候选，减少表格坐标和普通正文混在一起导致的召回失败。",
            "4. 混合召回：BM25 Top-80 + FAISS 向量 Top-40，经 RRF 融合后进入候选池。",
            "5. 规范识别与 metadata 排序：根据问题中的标准号、规范名、建筑场景、专项对象做通用/专项/旧版规范优先级处理。",
            "6. 本地重排：基于对象-属性匹配、条文号、表号、排除范围、废止替代提醒等通用规则调整排序。",
            "7. 回答层：DeepSeek 只基于召回证据生成结论；证据置信度和模型置信度分开展示，证据不足时不允许模型补写。",
            "",
            "## 4. 向量库与模型",
            "",
            f"- Embedding provider: `{vector_config.get('embedding_provider', '-')}`",
            f"- Embedding model: `{vector_config.get('embedding_model', '-')}`",
            f"- Index type: `{vector_config.get('index_type', '-')}`",
            f"- Chunks in vectorstore: `{vector_config.get('chunks', '-')}`",
            f"- Metric: `{vector_config.get('metric', '-')}`",
            "",
            "## 5. 评估结果",
            "",
            "### 5.1 聚焦单轮评估",
            "",
            f"- Cases: {focus_summary.get('cases', '-')}",
            f"- Scored cases: {focus_summary.get('scored_cases', '-')}",
            f"- Hit@1 / Hit@3 / Hit@5: {fmt_rate(focus_summary.get('hit_at_1'))} / {fmt_rate(focus_summary.get('hit_at_3'))} / {fmt_rate(focus_summary.get('hit_at_5'))}",
            f"- Standard@1 / Article@1: {fmt_rate(focus_summary.get('standard_at_1'))} / {fmt_rate(focus_summary.get('article_at_1'))}",
            f"- Field recall: `{json.dumps(focus_summary.get('field_recall_rate', {}), ensure_ascii=False)}`",
            "",
            "### 5.2 自然语言场景评估",
            "",
            f"- Cases: {natural_summary.get('cases', '-')}",
            f"- Single-turn / multi-turn: {natural_summary.get('single_turn_cases', '-')} / {natural_summary.get('multi_turn_cases', '-')}",
            f"- Scored single-turn cases: {natural_summary.get('scored_single_turn_cases', '-')}",
            f"- Hit@1 / Hit@3 / Hit@5: {fmt_rate(natural_summary.get('hit_at_1'))} / {fmt_rate(natural_summary.get('hit_at_3'))} / {fmt_rate(natural_summary.get('hit_at_5'))}",
            f"- Boundary pass rate: {fmt_rate(natural_summary.get('boundary_pass_rate'))}",
            f"- Multi-turn focus hit rate: {fmt_rate(natural_summary.get('multi_turn_focus_hit_rate'))}",
            f"- Multi-turn answerable rate: {fmt_rate(natural_summary.get('multi_turn_answerable_rate'))}",
            "",
            "### 5.3 多轮专项评估",
            "",
            f"- Cases / turns: {multiturn_summary.get('cases', '-')} / {multiturn_summary.get('turns', '-')}",
            f"- Retrieval Hit@5: {fmt_rate(multiturn_summary.get('retrieval_hit_at_5'))}",
            f"- Behavior hit rate: {fmt_rate(multiturn_summary.get('behavior_hit_rate'))}",
            f"- Low-confidence turns: {multiturn_summary.get('low_confidence_turns', '-')}",
            "",
            "## 6. 当前可信边界",
            "",
            "- 可以展示：标准号、条文号、PDF 物理页、原文片段、证据置信度、模型判断置信度、低置信拒答原因。",
            "- 不应过度承诺：地方审查口径、完整版本差异、未收录规范、没有 IFC 或项目条件时的合规结论。",
            "- 剩余主要风险：旧版设计规范与现行通用规范竞争时，Top1 可能更偏“产品正确性”而不是测试集预设；表格内精确数值字段抽取仍弱于标准/条文/page 召回。",
            "",
            "## 7. 下一步",
            "",
            "1. 扩到 100 题以上：补充更多表格坐标、跨规范冲突、地方/未收录边界、同对象不同属性问题。",
            "2. 增加语义 reranker A/B：对比本地重排与 `BAAI/bge-reranker-v2-m3` 的 Top1、MRR、成本和稳定性。",
            "3. 增加回答质量评估：不只看召回，还评估结论是否引用正确、是否拒绝越界问题、是否把场景和数值说清楚。",
            "4. IFC 集成前置条件：RAG 的引用和低置信机制稳定后，再把 IFC 解析出的建筑类型、楼层、房间、门宽、疏散距离等结构化字段送入问答/审查链路。",
        ]
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
