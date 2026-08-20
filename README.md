<img width="1293" height="913" alt="8b0bb1a0-618a-4070-9c41-6fc50574513d" src="https://github.com/user-attachments/assets/a1fe1a5f-cdea-4c78-9315-948d60785b7b" />
# 建筑消防规范审查系统 RAG

本项目是一套面向建筑消防规范审查场景的 RAG 原型系统。系统围绕消防规范检索、BIM/IFC 模型事实解析、证据展示和分层评估展开，用于辅助完成规范问答、模型字段核对和消防合规审查准备。

系统定位为审查辅助和方法验证工具，不替代正式审查流程、专业判断或主管部门审批。

## 核心能力

- 消防规范问答：检索相关条文，返回标准号、条文号、PDF 物理页、原文片段、证据置信度和模型判断置信度。
- BIM 模型审查：解析 IFC/IFCXML 文件，提取楼层、空间、门、楼梯、建筑高度等模型事实，支持模型事实问答和消防合规审查。
- 规范版本与适用性提示：维护轻量规则层，记录规范状态、施行日期和跨规范适用关系。
- 分层评估：包含 IFC 事实解析、RAG 检索、合规判定、多轮上下文、回答质量和业务提效测量的评估脚本与样例集。
- 系统观测：提供健康检查、请求指标、模型调用指标、向量检索后端状态和可选访问控制。

## 系统架构

```text
规范 PDF -> 文本抽取 -> 切块与元数据 -> BM25 + FAISS/Qdrant 混合检索
IFC 模型 -> IfcOpenShell 解析 -> 模型事实 -> 模型问答 / 合规审查
React 界面 -> FastAPI 服务 -> 检索、解析、回答生成、评估脚本
```

主要目录：

- `app/`：FastAPI 后端服务。
- `scripts/`：解析、检索、回答生成、评估和辅助脚本。
- `web/`：React/Vite 前端界面。
- `docs/`：产品、架构、评估和业务测量文档。
- `data/eval_sets/`：小型评估集和提效基准任务定义。
- `data/policy/`：规范目录、适用性和版本对比元数据。

## 数据共享边界

公开仓库不包含原始规范文件、生成后的 chunks、向量索引、上传的 IFC 文件或完整评估运行结果。这些内容可能包含较大的生成产物或源文档文本，应由使用者在本地按授权范围自行构建。

建议公开：

- 源代码。
- 配置模板。
- 规范策略元数据。
- 评估样例定义。
- 方法与架构文档。

不建议公开：

- `data/raw_pdfs/`
- `data/processed_text/`
- `data/vectorstore/`
- `data/structured_tables/`
- `data/ifc_uploads/`
- `data/retrieval_runs/`

## 本地启动

后端：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
.\.venv\Scripts\python.exe -m uvicorn app.api_server:app --host 127.0.0.1 --port 8010
```

前端：

```powershell
cd web
npm install
npm run dev -- --host 127.0.0.1 --port 8507
```

访问地址：

- 前端页面：`http://127.0.0.1:8507/`
- 后端健康检查：`http://127.0.0.1:8010/api/health`

## 环境变量

复制 `.env.example` 为 `.env`，并按本地或部署环境填写需要的变量。

- `DEEPSEEK_API_KEY`：用于回答生成和上下文路由。
- `SILICONFLOW_API_KEY`：用于 embedding 和可选重排。
- `APP_ACCESS_TOKEN`：可选接口访问令牌。
- `QDRANT_URL`、`QDRANT_COLLECTION`、`QDRANT_API_KEY`：可选 Qdrant 后端；未配置时使用 FAISS。

## 构建知识库

将规范 PDF 放入 `data/raw_pdfs/`，再执行本地构建脚本：

```powershell
.\.venv\Scripts\python.exe scripts\parse_and_chunk.py
.\.venv\Scripts\python.exe scripts\build_table_index.py
.\.venv\Scripts\python.exe scripts\build_faiss_vectorstore.py --env-file .env
```

生成产物默认不会进入 Git。

## 评估

单元测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s scripts -p "test_*.py"
```

前端构建：

```powershell
cd web
npm run build
```

分层评估示例：

```powershell
.\.venv\Scripts\python.exe scripts\eval_all.py --only ifc
.\.venv\Scripts\python.exe scripts\eval_all.py --only compliance rag multiturn answers --env-file .env
```

业务提效测量见 `docs/business_efficiency_evaluation.md`。只有在同一组任务下采集人工基线和系统辅助基线之后，才建议对外报告百分比提效。

## 许可证

如果希望他人在明确条款下复用或修改项目，发布前应补充许可证文件。

<img width="1293" height="913" alt="18a8c5f4-5e17-442f-95e9-8662009bbbf2" src="https://github.com/user-attachments/assets/7177550c-18ac-4c0a-9c3e-37edb3034e3a" />
<img width="1293" height="913" alt="8b0bb1a0-618a-4070-9c41-6fc50574513d" src="https://github.com/user-attachments/assets/55e6c184-e3b1-40c6-9058-bc914af34e25" />


