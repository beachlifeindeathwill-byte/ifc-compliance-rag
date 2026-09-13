# BIM Fire Code Review Agent

This project is a prototype workbench for building fire-code retrieval and BIM/IFC-assisted compliance review. It combines a RAG pipeline over building-code documents with IFC model parsing, evidence display, and layered evaluation.

The system is intended for review assistance and research. It does not replace formal code review, professional judgement, or authority approval.

## What It Does

- Fire-code Q&A: retrieves relevant clauses and returns standard id, article number, PDF page, source excerpts, evidence confidence, and model confidence.
- BIM model review: parses IFC/IFCXML files, extracts model facts, supports model-fact Q&A, and combines model facts with retrieved clauses for compliance review.
- Version comparison: maintains a lightweight policy layer for standard status, effective dates, and cross-standard applicability notes.
- Evaluation: includes test sets and scripts for IFC fact extraction, retrieval quality, compliance verdicts, multi-turn behavior, answer quality, and productivity measurement.
- Observability: exposes health, request metrics, LLM call metrics, vector backend status, and optional access control.

## Architecture

```text
PDF standards -> text extraction -> chunks/metadata -> BM25 + FAISS/Qdrant retrieval
IFC model -> IfcOpenShell parser -> model facts -> model Q&A / compliance review
React UI -> FastAPI -> retrieval, parsing, LLM answer generation, evaluation scripts
```

Key directories:

- `app/`: FastAPI server.
- `scripts/`: parsers, retrieval, answer generation, evaluation, and utility scripts.
- `web/`: React/Vite frontend.
- `docs/`: product, architecture, evaluation, and measurement documents.
- `data/eval_sets/`: compact evaluation sets and benchmark task definitions.
- `data/policy/`: standard registry, applicability, and version-comparison metadata.

## Data Policy

The public repository does not include raw standards, generated chunks, vector indexes, uploaded IFC files, or full evaluation run outputs. These files can contain large generated artifacts or source-document text and should be rebuilt locally.

Recommended public contents:

- Source code.
- Configuration templates.
- Policy metadata.
- Evaluation case definitions.
- Summary documentation.

Not recommended for public sharing:

- `data/raw_pdfs/`
- `data/processed_text/`
- `data/vectorstore/`
- `data/structured_tables/`
- `data/ifc_uploads/`
- `data/retrieval_runs/`

## Setup

Backend:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
.\.venv\Scripts\python.exe -m uvicorn app.api_server:app --host 127.0.0.1 --port 8010
```

Frontend:

```powershell
cd web
npm install
npm run dev -- --host 127.0.0.1 --port 8507
```

Open:

- Frontend: `http://127.0.0.1:8507/`
- API health: `http://127.0.0.1:8010/api/health`

## Environment Variables

Copy `.env.example` to `.env` and configure the values needed by your deployment.

- `DEEPSEEK_API_KEY`: LLM answer generation and context routing.
- `SILICONFLOW_API_KEY`: embedding and optional reranking provider.
- `APP_ACCESS_TOKEN`: optional API access token.
- `QDRANT_URL`, `QDRANT_COLLECTION`, `QDRANT_API_KEY`: optional Qdrant backend. If unset, FAISS is used.

## Build Knowledge Artifacts

Place source PDFs under `data/raw_pdfs/`, then run the local build scripts:

```powershell
.\.venv\Scripts\python.exe scripts\parse_and_chunk.py
.\.venv\Scripts\python.exe scripts\build_table_index.py
.\.venv\Scripts\python.exe scripts\build_faiss_vectorstore.py --env-file .env
```

Generated artifacts are ignored by Git by default.

## Evaluation

Unit tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s scripts -p "test_*.py"
```

Frontend build:

```powershell
cd web
npm run build
npm run test:sites
```

Layered evaluation examples:

```powershell
.\.venv\Scripts\python.exe scripts\eval_all.py --only ifc
.\.venv\Scripts\python.exe scripts\eval_all.py --only compliance rag multiturn answers --env-file .env
```

Business-impact measurement is described in `docs/business_efficiency_evaluation.md`. Report percentage improvement only after collecting a manual baseline and an assisted baseline with the same task definitions.

## License

Add a license file before publishing if you want others to reuse or modify the project under explicit terms.
