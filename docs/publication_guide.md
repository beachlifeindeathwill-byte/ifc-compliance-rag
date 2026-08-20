# 公开发布说明

本文档说明建筑消防规范审查系统 RAG 在公开仓库中建议包含的内容，以及应保留在本地的数据边界。

## 建议包含

- `app/`、`scripts/` 和 `web/` 中的应用源代码。
- `.env.example` 等公开配置模板。
- `docs/` 中的架构、评估和产品说明文档。
- `data/eval_sets/` 中的小型评估集。
- `data/policy/` 中的规范目录和适用性元数据。

## 不建议包含

- 原始规范文件和源 PDF。
- 生成后的 chunks、抽取页、表格索引、向量库、embedding 缓存和完整检索运行结果。
- 用户上传的 IFC 模型。
- 本地截图、视觉检查记录、日志和临时文件。
- 密钥、API Key、本地绝对路径和个人环境说明。

## Chunk 共享策略

默认不发布生成后的 chunks。chunk 文件通常包含较多源文档文本，容易被视为重新分发规范原文。建议公开：

- 切块脚本；
- chunk 数据结构；
- 必要时提供短小的合成样例；
- 从授权文档本地重建 chunks 的说明。

## 发布检查清单

1. 确认 `.env` 未被 Git 跟踪。
2. 确认 `data/raw_pdfs/`、`data/processed_text/`、`data/vectorstore/`、`data/structured_tables/`、`data/ifc_uploads/` 和 `data/retrieval_runs/` 未被 Git 跟踪。
3. 运行后端单元测试。
4. 运行前端构建。
5. 检查 README 中是否存在本地路径或私有表达。
6. 为公开版本创建 Git tag，便于后续多轮迭代。
