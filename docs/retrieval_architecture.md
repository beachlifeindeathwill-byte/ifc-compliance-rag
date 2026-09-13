# 检索架构与比例

## 当前真实状态

当前 v2 已补充 FAISS 构建脚本，并已落盘正式 SiliconFlow `BAAI/bge-m3` FAISS 向量库。

现有存储是：

- 原始 PDF：`data/raw_pdfs/`
- 结构化切片：`data/processed_text/chunks.jsonl`
- 人工抽检文件：`data/chunks_audit/`
- 评估集：`data/eval_sets/fire_code_eval_v2.jsonl`
- 规范关系配置：`data/policy/`

当前检索评估实际使用的是：

1. `chunks.jsonl` 加载到内存
2. BM25/关键词检索
3. 规范识别层
4. metadata 加权
5. 废止条文 seed 标记
6. 本地 reranker v0

当前 `data/vectorstore/faiss.index` 存在，配置为 `embedding_provider=siliconflow`、`embedding_model=BAAI/bge-m3`，共 2330 个 chunk、1024 维向量。

## FAISS 构建命令

```powershell
python scripts/build_faiss_vectorstore.py --env-file .env
```

成功后会生成：

- `data/vectorstore/faiss.index`
- `data/vectorstore/embeddings.npy`
- `data/vectorstore/chunk_meta.jsonl`
- `data/vectorstore/embedding_config.json`

如果 SiliconFlow embedding 接口临时不可达，可先构建本地 hash FAISS fallback：

```powershell
python scripts/build_faiss_vectorstore.py --provider local-hash
```

注意：`local-hash` 只是应急方案。当前正式库已经使用 SiliconFlow `BAAI/bge-m3` 重建。

混合检索评估命令：

```powershell
python scripts/evaluate_hybrid_retrieval.py --env-file .env
```

检索可视化 UI 启动命令：

```powershell
python -m uvicorn app.api_server:app --host 127.0.0.1 --port 8010
```

## 计划中的混合检索比例

正式问答建议使用混合检索：

| 阶段 | 策略 | 数量 |
| --- | --- | ---: |
| 第一召回 | BM25 精确召回 | Top 80 |
| 第一召回 | 向量语义召回 | Top 40 |
| 融合 | BM25 主导 + RRF 辅助 | 合并到约 Top 80 |
| 规范策略 | 标准路由、metadata 加权、废止条文标记 | Top 30 |
| Rerank | SiliconFlow rerank 或本地 reranker | Top 8 |
| 回答上下文 | 最终喂给大模型 | Top 5 |

当前实现中，第一阶段采用 `BM25 : Vector = 2 : 1` 的偏精确检索比例。

原因：

- BM25 适合规范标准号、条文号、数值、专有名词，例如 `5.5.18`、`0.80m`、`消防车道净宽度`。
- 向量检索适合用户口语化问题，例如“车库要不要喷淋”“疏散门最窄能多宽”。
- 消防规范问答不能只靠向量相似度，否则容易被其他规范中相似关键词吸走。实际评估中，BM25 主导的混合排序比 1:1 RRF 更稳。

## Rerank API 接法

SiliconFlow 官方 rerank 接口：

```text
POST https://api.siliconflow.cn/v1/rerank
```

推荐先用：

```text
BAAI/bge-reranker-v2-m3
```

请求结构：

```json
{
  "model": "BAAI/bge-reranker-v2-m3",
  "query": "用户问题",
  "documents": ["候选chunk 1", "候选chunk 2"],
  "return_documents": false,
  "top_n": 8
}
```

鉴权继续使用：

```text
SILICONFLOW_API_KEY
```

一般不需要重新申请新的 API 口。如果已经有 SiliconFlow key，只需要确认：

1. 账号余额可用。
2. 账号对 rerank 模型有访问权限。
3. 当前模型列表里仍提供目标 reranker。
4. 环境变量 `SILICONFLOW_API_KEY` 已配置。

当前已封装 `scripts/siliconflow_rerank.py`，也已在 hybrid 检索中提供 `use_api_rerank` 可选开关，但默认不启用。原因是中文小样本测试中，`BAAI/bge-reranker-v2-m3` 出现了明显排序异常，并且返回文档中的中文被回显为问号。正式接入前必须单独做 reranker A/B 评估。

## 产品风险控制

最终回答不能只展示模型生成文本，必须展示：

1. 命中的标准名称
2. 条文号和页码
3. rerank 分数
4. 规范路由原因
5. 是否命中废止条文 seed 表
6. 最终采用哪份规范以及原因

这样可以证明系统不是“PDF 语义搜索壳”，而是考虑了法规适用性和冲突处理。
