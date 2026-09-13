# 第一轮规范解析与切片报告

## 本轮结论

已完成 4 份 PDF 的文本画像，并对 4 份可提取文本的规范生成结构化 chunk。

`GB 55036-2022.pdf` 已替换为可提取文本版本，并纳入本轮切片。

## 输出文件

| 文件 | 说明 |
| --- | --- |
| `data/processed_text/pdf_profiles.json` | PDF 文本质量画像 |
| `data/processed_text/chunks.jsonl` | 全量结构化 chunk |
| `data/processed_text/chunk_summary.json` | 切片统计摘要 |
| `data/processed_text/skipped_docs.json` | 跳过文档及原因 |
| `data/chunks_audit/` | 人工抽检用 chunk 文本 |
| `scripts/parse_and_chunk.py` | 解析和切片脚本 |
| `docs/retrieval_evaluation.md` | 第一版检索评估结论 |

## 切片统计

| 标准编号 | chunk 数 | 含条文号 chunk | 含表格 chunk | 处理状态 |
| --- | ---: | ---: | ---: | --- |
| GB 50016-2014 | 1454 | 1030 | 123 | 已切片，需重点抽检表格 |
| GB 50067-2014 | 183 | 150 | 8 | 已切片 |
| GB 55037-2022 | 366 | 307 | 4 | 已切片 |
| GB 55036-2022 | 327 | 263 | 0 | 已切片 |

## 初步抽检发现

1. `GB 50016-2014` 中 `5.5.18` 已被切成独立 chunk，并保留了高层公共建筑疏散门、疏散走道、疏散楼梯最小净宽度表。
2. 消防车道、防火分区、自动喷水灭火系统、火灾自动报警系统、灭火器、汽车库、疏散楼梯等关键词均能在 chunk 中找到候选片段。
3. 部分 PDF 提取文本存在识别瑕疵，例如：
   - `不应小于` 被提取为 `不应小子`
   - `0.90m` 个别位置被提取为 `0.9001`
   - 个别表格列名有错字或错位

这些问题需要在人工抽检中记录。后续回答生成时不能只依赖模型自信表达，必须展示原文片段供复核。

## 下一步

1. 从 `data/chunks_audit/` 中抽检高频条文 chunk。
2. 人工复核 `data/eval_sets/fire_code_eval_v1.jsonl` 中的 expected 标准和条文。
3. 增加规范识别、metadata 加权和 reranker，对比 Hit@1/Hit@5 是否提升。
4. 生成 embedding 向量库，并跑语义检索 Top-K 召回评估。
5. 抽检和评估通过后，再接入独立规范问答界面。
