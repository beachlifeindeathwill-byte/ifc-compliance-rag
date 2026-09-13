# 检索评估报告

## 当前版本

- 评估集：`data/eval_sets/fire_code_eval_v2.jsonl`
- 问题数：60
- 规范范围：GB 50016-2014、GB 50067-2014、GB 55036-2022、GB 55037-2022
- 检索版本：BM25/关键词基线 + policy-aware 规范路由 + 本地 reranker v0
- 输出文件：`data/retrieval_runs/retrieval_eval_v1.*`、`data/retrieval_runs/retrieval_eval_v2_policy.*`

## 指标

| 指标 | BM25 基线 | Policy-aware + reranker v0 | Hybrid FAISS bge-m3 |
| --- | ---: | ---: | ---: |
| Hit@1 | 0.700 | 0.767 | 0.817 |
| Hit@3 | 0.817 | 0.933 | 0.933 |
| Hit@5 | 0.850 | 0.933 | 0.933 |
| MRR | 0.763 | 0.847 | 0.880 |
| Standard@1 | 未单独输出 | 0.967 | 0.967 |
| Standard@3 | 未单独输出 | 1.000 | 1.000 |
| Standard@5 | 未单独输出 | 1.000 | 1.000 |
| Top-1 deprecated seed rate | 未单独输出 | 0.250 | 未单独输出 |

## 焦点测试

- 测试集：`data/eval_sets/fire_code_focus_single_turn.jsonl`
- 用例数：15
- 可判分用例：13
- Hit@1：0.769
- Hit@3：1.000
- Hit@5：1.000
- 边界题降置信度命中：2/2
- 说明：该测试集覆盖表格坐标、页码直达、数值判断、边界问题，难度高于普通 60 题集。当前已能识别版本差异、未指定标准的页码定位等不可直接回答场景；剩余弱点主要是少量跨条款综合题和通用规范/旧版详细条文之间的最终适用性判断。

## 多轮测试

- 测试集：`data/eval_sets/fire_code_multiturn_cases.json`
- 用例数：4 组
- 回合数：15
- 检索断言数：6
- Retrieval Hit@5：1.000
- 行为断言数：5
- Behavior hit rate：1.000
- 低置信度回合数：12
- 输出文件：`data/retrieval_runs/multiturn_eval.json`、`data/retrieval_runs/multiturn_eval.md`
- 说明：当前多轮能力采用轻量上下文重写器，将“那、这个、刚才、哪一条”等追问补全为可检索问题，并在 UI 中提供“连续追问模式”。这不是完整的长期记忆或语义摘要记忆；当前回答层仍保持保守，证据排序不稳或条件不足时会降置信度。

## 结论

当前切片已经能支撑基础召回，加入规范路由、metadata 加权、废止条文 seed 标记和本地 reranker v0 后，文档级适用性明显提升。

当前已落盘正式 FAISS，使用 SiliconFlow `BAAI/bge-m3` 生成 2330 个 chunk 的 1024 维向量。Hybrid 检索采用 BM25 Top 80 + FAISS Top 40，BM25 精确分主导，FAISS 语义分作为补充，再进入规范路由、本地 reranker v0 和结构化表格候选补充。

当前主要剩余问题不是“被错误规范吸走”，而是少量抽象问题仍需要跨条款综合和人工确认最终适用性。当前已增加检索可视化 UI、连续追问模式，并提供实验性 SiliconFlow API rerank 开关；默认仍使用本地 reranker v0，正式启用 API reranker 前需要单独做 A/B 评估。

## v1 失败与弱召回样例

| 问题 | 现象 | 初步判断 |
| --- | --- | --- |
| Q006 建筑高度大于100米的公共建筑是否应设置避难层？ | 正确条文 GB 50016-2014 5.5.23 排第 5 | 相关解释性段落和住宅避难层条文竞争，需要条文级优先和精排 |
| Q020 建筑防火通用规范对建筑防火的基本目标是什么？ | 未命中 GB 55037-2022 1.0.x | 问题含规范标题，但基础 BM25 没有利用标准标题 metadata |
| Q021 耐火等级和构件耐火极限通用要求 | 未命中 GB 55037-2022 3.x | 老规范 GB 50016 的同类内容更长、更高频，需要规范优先级和 reranker |
| Q022 防火分隔通用要求 | 未命中 GB 55037-2022 4.x | 查询过宽，需先做适用规范识别或 query rewrite |
| Q023 安全疏散和避难通用要求 | 未命中 GB 55037-2022 7.x | GB 50016 解释性内容丰富，压过通用规范 |
| Q024 消防救援设施通用要求 | 正确结果排第 6 | 接近命中，但 Top-5 不稳定，需要 reranker 或 metadata 加权 |

## 下一轮改进

1. 已完成：增加规范识别层，先判断问题更像“建筑防火通用规范”“消防设施通用规范”“车库专项规范”还是旧版设计规范。
2. 已完成：增加 metadata 排序因子，当问题明确出现规范名称、标准号、通用规范、专项规范时，优先对应标准。
3. 已完成：增加废止条文标记，旧规范条文如果已被通用规范废止，不能直接作为最终答案。
4. 部分完成：增加本地 reranker v0，并预留 SiliconFlow API rerank 开关；API rerank 默认关闭，待 A/B 评估。
5. 已完成：增加检索可视化，在界面展示命中的标准、页码、条文号、分数、原文片段和排序原因。
6. 部分完成：评估集已从 24 题扩展到 60 题，后续可继续扩到 100 题，覆盖更多冲突和边界场景。
7. 已完成：增加通用问题边界层和直接定位入口，识别版本差异、页码定位、条文号定位和表号定位；证据不足时降置信度，不让大模型补写答案。
8. 已完成：增加多轮追问评估和 UI 连续追问模式；当前是上下文重写，不是长期语义记忆，后续可继续做摘要记忆和用户确认机制。

## v2 Policy-Aware 补充

已新增 `data/policy/standard_registry.json` 和 `data/policy/abolished_articles_seed.json`，并通过 `scripts/evaluate_policy_retrieval.py` 生成带规范路由和废止条文提示的评估结果。

评估集已从 `fire_code_eval_v1.jsonl` 的 24 题扩展到 `fire_code_eval_v2.jsonl` 的 60 题，覆盖：

- 旧版建筑设计防火规范细节题
- 汽车库专项规范题
- 消防设施通用规范题
- 建筑防火通用规范题
- 新旧规范容易冲突的路由题

对应输出：

- `data/retrieval_runs/retrieval_eval_v2_policy.json`
- `data/retrieval_runs/retrieval_eval_v2_policy.md`
- `data/retrieval_runs/retrieval_eval_v3_hybrid.json`
- `data/retrieval_runs/retrieval_eval_v3_hybrid.md`

v2 额外关注 `Standard@K`，用于判断问题是否被错误规范吸走。`Hit@K` 关注条文级命中，`Standard@K` 关注文档级适用性。

已增加本地 reranker v0：对前言、目录、无条文号片段、条文说明续段降权，对正文条文片段、问题核心短语、数值下限和条款字段命中加权。该层用于减少 Top-30 候选中的词频噪声；同时已预留真正语义 reranker 的 API 开关，但默认不启用。

检索架构和后续向量库比例见 `docs/retrieval_architecture.md`。当前已建立正式 bge-m3 FAISS 向量库；当前 hybrid 实现使用 BM25 Top 80 + Vector Top 40，使精确条文检索主导、向量语义召回补充。当前还新增了通用结构化表格索引、direct locator 和多轮上下文重写，用于显式表号、页码、条文号、最大/最小面积、净宽、耐火等级、追问指代等问题；这些索引和重写规则按 metadata、字段重叠、建筑类别匹配、数值候选和历史上下文排序，不写入具体测试题答案。

## 对外表达口径

第一版不是直接追求回答生成，而是先把知识库建设拆成可验证流程：PDF 画像、结构化切片、人工抽检、评估集、Top-K 召回指标。评估集已从 24 题扩展到 60 题，并修正了“关键词命中但条文号不对也算 hit”的评估漏洞；在 BM25 基线上 Hit@5 为 85.0%，加入规范识别、metadata 加权、废止条文 seed 标记、本地 reranker v0、结构化表格索引和 direct locator 后，Hit@5 保持 93.3%，Standard@5 达到 100%。正式 bge-m3 FAISS 已落盘，hybrid 检索在当前权重下达到 Hit@1 81.7%、Hit@5 93.3%、Standard@5 100%；更难的 15 题焦点集达到 Hit@1 76.9%、Hit@3 100%、Hit@5 100%，并能把 2 个边界题正确降为低置信度。当前新增 4 组 15 回合多轮评估，检索断言 Hit@5 为 100%，行为断言命中率为 100%，但 12 个回合仍被标为低置信度，说明系统更偏审查型证据召回而不是强行生成答案。
