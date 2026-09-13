# BIM 消防合规 Agent 评估框架

本文档把项目从“聊天 RAG 手动测试”升级为可复现的分层评估。当前已覆盖 IFC 事实层、RAG 检索层、合规判定层、多轮行为和少量回答质量回归；业务提效需要另行通过人工基线对照实验测量，不能由技术指标直接换算。

## 分层状态

| 层级 | 当前状态 | 已落地指标 | 未落地指标 |
| --- | --- | --- | --- |
| L1 IFC 解析 | 已落地 | Entity Exact Match、Property Numeric List Match、Relation Set F1 | IFC 文件合法性校验、窗口/空间关系覆盖更多模型 |
| L2 Tool Calling | 未落地 | - | Function Accuracy、Argument Accuracy、Tool Call F1、Execution Success |
| L3 RAG 检索 | 已部分落地 | Hit@1/3/5、Standard@、Article@、Precision@1/3/5、MRR@10、字段召回 | Context Precision、Context Recall、Faithfulness |
| L4 合规判定 | 已落地 | Verdict Accuracy、Safety False Negative Rate、三类结论覆盖 | Macro-F1、更大规模真实案例 |
| L5 最终回答 | 基础回归已落地 | 置信度、缺失字段提示、Required Claim Pass、Quality Audit Pass | Faithfulness、Citation Correctness、Answer Relevancy |
| L6 Agent 行为 | 已有基础，待升级 | 多轮检索命中、行为断言、边界拒答 | Entity Tracking、Context Retention、Boundary Success |
| L7 业务提效 | 待测 | 任务计时方案、提效计算脚本 | 真实用户或模拟用户基线 |

## 报告口径

`data/retrieval_runs/eval_all_report.md` 中如果某一层显示 `not_run`，表示该层没有在本轮 `eval_all.py` 执行，只展示已有历史结果用于参考；不能把这类行描述为“本轮全量通过”。对外表达时应同时说明评估日期、样本规模和是否为本轮执行。

## 与已有三层级测试的关系

项目之前已经存在“单轮、多轮、边界”三类测试，它们不冲突，只是当前六层框架的更上层表达。

| 旧测试类别 | 已有资产 | 在六层框架中的位置 |
| --- | --- | --- |
| 单轮对话 | `fire_code_natural_scenarios` 简单问题、`fire_code_focus_single_turn`、`fire_code_answer_regression` A | L3 RAG 检索 + L5 回答质量 |
| 多轮对话 | `fire_code_natural_scenarios` C001-C005、`fire_code_multiturn_cases`、`fire_code_answer_regression` B | L6 Agent 行为，后续升级为实体追踪和状态切换 |
| 边界测试 | `fire_code_natural_scenarios` J006-J008/J012、聚焦单轮 boundary、`fire_code_answer_regression` C | L6 边界 + L4 信息不足判定 |

旧的单轮/多轮/边界测试重点验证“用户看到的回答对不对”；新的 IFC、Tool、Compliance、Faithfulness 测试重点回答“如果错了，是解析层、工具层、检索层还是推理层错了”。后续不应删除旧资产，而应把旧场景逐步迁移成新框架中的 case，例如把 J001-J005 迁移到 `compliance_cases.jsonl`，把 C001-C005 升级为带 entity tracking 的多轮场景。

## L1 IFC 事实评估

生成 oracle 测试集：

```powershell
python scripts/build_ifc_eval_set.py
```

运行解析器对比：

```powershell
python scripts/evaluate_ifc_facts.py
```

最近一次样例结果：

- Entity Accuracy: 100%
- Property Accuracy: 100%
- Relation Accuracy: 100%
- Derived Accuracy: 100%

测试集从 IfcOpenShell 直接读取实体数量、门宽、窗宽、空间/门所属楼层和楼层高程，不把运行时规则当作 Ground Truth。

## L3 RAG 检索评估

运行聚焦单轮评估：

```powershell
python scripts/evaluate_focus_tests.py --env-file .env
```

最近一次样例结果：

- Hit@1 / Hit@3 / Hit@5: 98.7% / 100% / 100%
- Precision@1 / @3 / @5: 98.7% / 48.2% / 31.8%
- MRR@10: 99.3%
- 数值或关键字段召回: 80%

Precision@3 和 @5 明显低于 Hit@K，说明正确条文虽然能进入 Top-K，但候选集中仍然混入较多不相关条文。后续应把优化目标从“Top-1 是否命中”扩展到“Top-K 是否保持高相关性”。

## L4 合规判定评估

运行最小合规评估：

```powershell
python scripts/evaluate_compliance_cases.py --env-file .env
```

最近一次样例结果：

- Verdict Accuracy: 100%（25 例历史回归）
- Safety False Negative Rate: 0%
- 覆盖 `PASS / FAIL / INSUFFICIENT_INFORMATION` 三类输出
- 覆盖门宽、防火分区面积、疏散距离、疏散楼梯净宽、中庭防火卷帘、信息缺失、数据冲突等场景

合规接口已返回结构化 `verdict`，并保留 `compliance_reasons`、`critical_risk` 和 `missing_fields`。历史回归中安全漏检为 0，但仍需要继续扩大真实案例和多模型样本；不能只用 25 例合规用例代表生产环境效果。

## L7 业务提效评估

业务提效采用人工基线对照实验，不从 Hit@1、Verdict Accuracy 等技术指标直接推导。任务、记录字段和计算口径见 `docs/business_efficiency_evaluation.md`。

运行提效汇总：

```powershell
python scripts/evaluate_productivity_benchmark.py data/manual/productivity_benchmark.csv
```

在没有真实计时数据前，只能描述“建立了可测量的评估方案”，不能写“提效 X%”。

## 下一步

1. L2：把 IFC 模型查询从“解析摘要直接交给 LLM”改成显式工具调用，并记录工具名、参数和执行结果，才能计算 Tool Accuracy。
2. L4：把面积、疏散距离、防火分区等数值判断迁移到确定性规则引擎，消除 1800㎡/2000㎡ 这类简单算术判断的偶发错误。
3. L5：用规则或 LLM Judge 检查回答引用的条文是否真的来自召回证据，计算 Faithfulness 和 Citation Correctness。
4. L6：把多轮用例扩展为实体追踪、对象切换、长距离回溯和五类边界问题。
5. 已增加 `eval_all.py`，支持 `--only ifc/compliance/rag/multiturn` 选择子集，统一输出各层指标并保存 baseline。
6. 执行 `docs/business_efficiency_evaluation.md` 中的 5 个任务计时，补齐业务影响证据。
