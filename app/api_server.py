from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from conversation_context import (
    ConversationState,
    merge_evidence_results,
    normalize_reference_context,
    rank_context_evidence,
    recent_evidence,
    remember_evidence,
    remember_question,
)
from context_router import ContextDecision, route_context_with_llm
from hybrid_retrieval import hybrid_search, load_vectorstore
from ifc_fire_parser import build_ifc_review_questions, ifc_processing_chain, parse_ifc_file
from llm_answer import LLMAnswerError, generate_answer
from observability import now_ms, record_http_request, record_llm_usage, snapshot_metrics
from project_profile import apply_project_conflict_guard, compose_compliance_context, normalize_project_profile, project_preparation_prompt
from qdrant_retriever import qdrant_enabled
from retrieval_policy import load_policy
from standard_timeline import get_version_topic, list_version_topics, load_standard_registry, resolve_version_question


IFC_UPLOAD_DIR = ROOT / "data" / "ifc_uploads"
RAW_PDF_DIR = ROOT / "data" / "raw_pdfs"
ENV_FILES = [ROOT / ".env"]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


for env_file in ENV_FILES:
    load_env_file(env_file)
IFC_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Building Fire Review API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8507", "http://localhost:8507"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ACCESS_TOKEN = os.getenv("APP_ACCESS_TOKEN", "")


@app.middleware("http")
async def gateway(request, call_next):
    started = now_ms()
    try:
        if ACCESS_TOKEN and request.url.path != "/api/health":
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {ACCESS_TOKEN}":
                return JSONResponse(status_code=401, content={"detail": "需要访问令牌"})
        response = await call_next(request)
        record_http_request(request.url.path, now_ms() - started, response.status_code < 500)
        return response
    except Exception:
        record_http_request(request.url.path, now_ms() - started, False)
        raise


class HistoryTurn(BaseModel):
    question: str
    answer: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    sources: list[dict[str, Any]] = Field(default_factory=list)


class QARequest(BaseModel):
    question: str
    scope: str | None = None
    history: list[HistoryTurn] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=3, le=10)
    use_reranker: bool = False
    applicable_date: str = ""


class IFCQuestionRequest(BaseModel):
    summary: dict[str, Any]
    question: str
    project_context: str = ""
    project_profile: dict[str, Any] = Field(default_factory=dict)
    scope: str | None = None
    top_k: int = Field(default=5, ge=3, le=10)
    use_reranker: bool = False
    applicable_date: str = ""


class IFCReviewPreparationRequest(BaseModel):
    summary: dict[str, Any]
    project_context: str = ""


class VersionResolveRequest(BaseModel):
    question: str
    top_k: int = Field(default=8, ge=1, le=20)
    applicable_date: str = ""


def compact_text(value: Any, limit: int = 1600) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def public_source(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": item.get("chunk_id"),
        "source_file": item.get("source_file") or item.get("standard_id") or "未识别文件",
        "standard_id": item.get("standard_id") or "未识别",
        "article_no": item.get("article_no") or "未识别",
        "table_no": item.get("table_no"),
        "page": item.get("page"),
        "text": compact_text(item.get("text"), 1800),
        "has_table": bool(item.get("has_table")),
        "chunk_type": item.get("chunk_type"),
        "score": round(float(item.get("api_rerank_score") or item.get("hybrid_score") or 0.0), 3),
        "notes": [compact_text(note, 180) for note in item.get("policy_notes") or []] + ([compact_text(item["version_note"], 220)] if item.get("version_note") else []),
        "version_applicable": item.get("version_applicable"),
    }


def annotate_version_applicability(results: list[dict[str, Any]], applicable_date: str) -> list[dict[str, Any]]:
    if not applicable_date:
        return results
    registry = load_standard_registry()
    for item in results:
        standard_id = item.get("standard_id") or ""
        effective_date = registry.get("standards", {}).get(standard_id, {}).get("effective_date") or ""
        if not effective_date:
            continue
        item["version_applicable"] = applicable_date >= effective_date
        if applicable_date < effective_date:
            item["version_note"] = f"{standard_id} 施行日期 {effective_date} 晚于项目日期 {applicable_date}，不能直接作为适用依据，需确认项目时间口径"
        else:
            item["version_note"] = f"{standard_id} 施行日期 {effective_date} 不晚于项目日期 {applicable_date}"
    return results


def compact_ifc_facts(summary: dict[str, Any], *, max_items: int = 30) -> dict[str, Any]:
    counts = summary.get("counts") or {}
    return {
        "文件": summary.get("source_file"),
        "项目": summary.get("project"),
        "楼层数": counts.get("IfcBuildingStorey"),
        "空间数": counts.get("IfcSpace"),
        "门数量": counts.get("IfcDoor"),
        "楼梯数量": counts.get("IfcStair"),
        "建筑高度_m": summary.get("building_height_m"),
        "建筑用途推断": summary.get("building_use"),
        "可审查字段": summary.get("field_quality"),
        "楼层": [
            {"name": item.get("name"), "elevation_m": item.get("elevation_m")}
            for item in (summary.get("storeys") or [])[:max_items]
        ],
        "门": [
            {
                "name": item.get("name"),
                "storey": item.get("storey"),
                "width_m": item.get("width_m"),
                "height_m": item.get("height_m"),
                "fire_rating": item.get("fire_rating"),
                "is_exit_like": item.get("is_exit_like"),
            }
            for item in (summary.get("doors") or [])[:max_items]
        ],
        "空间": [
            {"name": item.get("name"), "storey": item.get("storey"), "area": item.get("area")}
            for item in (summary.get("spaces") or [])[:max_items]
        ],
    }


def evidence_assessment(results: list[dict[str, Any]], answer: dict[str, Any] | None = None) -> dict[str, str]:
    if not results:
        return {"level": "low", "reason": "未召回到候选原文。"}
    if answer:
        certainty = answer.get("certainty")
        missing_fields = answer.get("missing_fields") or []
        if certainty == "low" or not answer.get("can_answer"):
            return {"level": "low", "reason": "虽召回候选原文，但证据未完整覆盖问题所需的适用条件或关键字段。"}
        if certainty == "medium" or missing_fields:
            return {"level": "medium", "reason": "已召回相关原文，但适用条件或关键字段仍需复核。"}
    top = results[0]
    complete = all([top.get("standard_id"), top.get("article_no"), top.get("page") is not None, top.get("text")])
    risky = any("abolish" in str(note).lower() or "replacement" in str(note).lower() for note in top.get("policy_notes") or [])
    if complete and not risky:
        return {"level": "high", "reason": "首条证据包含标准号、条文号、PDF 物理页和完整原文片段。"}
    if top.get("text"):
        missing = []
        if not top.get("article_no"):
            missing.append("条文号")
        if top.get("page") is None:
            missing.append("PDF 页码")
        reason = "已召回相关原文"
        if missing:
            reason += f"，但缺少{'、'.join(missing)}"
        if risky:
            reason += "，且存在废止或替代复核提示"
        return {"level": "medium", "reason": reason + "。"}
    return {"level": "low", "reason": "候选结果缺少可核查的原文。"}


def apply_scope(results: list[dict[str, Any]], scope: str | None, top_k: int) -> list[dict[str, Any]]:
    if not scope or scope in {"auto", "all"}:
        return results[:top_k]
    scoped = [item for item in results if item.get("standard_id") == scope or item.get("source_file") == scope]
    return scoped[:top_k]


def build_state(history: list[HistoryTurn]) -> ConversationState:
    state = ConversationState()
    for turn in history[-6:]:
        remember_question(state, turn.question, turn.question)
        if turn.sources:
            remember_evidence(state, turn.sources)
    return state


def compact_turn_history(history: list[HistoryTurn]) -> list[dict[str, Any]]:
    return [
        {
            "question": turn.question,
            "conclusion": compact_text(turn.answer.get("conclusion"), 320),
            "sources": turn.sources[:3],
        }
        for turn in history[-6:]
    ]


def normalize_context_decision(
    question: str,
    decision: ContextDecision,
    state: ConversationState | None = None,
) -> ContextDecision:
    return normalize_reference_context(question, decision, state)


def select_turn_evidence(
    decision: ContextDecision | None,
    fresh_results: list[dict[str, Any]],
    context_results: list[dict[str, Any]],
    *,
    top_k: int,
    question: str,
) -> list[dict[str, Any]]:
    if not decision or not decision.use_previous_context or not context_results:
        return fresh_results[:top_k]
    if decision.mode == "evidence_request":
        ranked_context = rank_context_evidence(question, context_results)
        return merge_evidence_results(ranked_context, fresh_results, max_items=top_k)
    if decision.mode == "comparison":
        fresh_quota = max(1, (top_k + 1) // 2)
        context_quota = max(1, top_k - fresh_quota)
        selected = merge_evidence_results(
            fresh_results[:fresh_quota],
            context_results[:context_quota],
            max_items=top_k,
        )
        if len(selected) < top_k:
            selected = merge_evidence_results(selected, fresh_results + context_results, max_items=top_k)
        return selected
    # Ordinary follow-ups inherit semantic scope through the rewritten query. Old evidence
    # must not outrank fresh evidence when the user changes the reviewed property.
    return fresh_results[:top_k]


def retrieve(
    question: str,
    *,
    scope: str | None,
    top_k: int,
    use_reranker: bool,
    candidate_top_k: int | None = None,
) -> list[dict[str, Any]]:
    final_k = max(top_k, candidate_top_k or top_k)
    results = hybrid_search(
        question,
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        use_api_rerank=use_reranker,
        rerank_api_key=os.getenv("SILICONFLOW_API_KEY"),
        rerank_top_n=final_k,
        final_top_k=max(20 if scope not in {None, "", "auto", "all"} else final_k, final_k),
    )
    return apply_scope(results, scope, final_k)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "deepseek": bool(os.getenv("DEEPSEEK_API_KEY")), "reranker": bool(os.getenv("SILICONFLOW_API_KEY"))}


@app.get("/api/status/metrics")
def status_metrics() -> dict[str, Any]:
    return {
        "metrics": snapshot_metrics(),
        "access_control": bool(ACCESS_TOKEN),
        "vector_backend": "qdrant" if qdrant_enabled() else "faiss",
    }


@app.get("/api/library")
def library() -> dict[str, Any]:
    registry, _ = load_policy()
    _, metas, vector_config = load_vectorstore()
    counts: dict[str, int] = {}
    files: dict[str, str] = {}
    for meta in metas:
        standard_id = meta.get("standard_id") or "未识别"
        counts[standard_id] = counts.get(standard_id, 0) + 1
        if meta.get("source_file"):
            files[standard_id] = meta["source_file"]
    standards = []
    for standard_id, config in registry.get("standards", {}).items():
        standards.append(
            {
                "standard_id": standard_id,
                "title": config.get("title") or standard_id,
                "source_file": files.get(standard_id) or config.get("file") or f"{standard_id}.pdf",
                "chunks": counts.get(standard_id, 0),
                "status": "现行有效",
            }
        )
    known = {item["source_file"] for item in standards}
    for path in sorted(RAW_PDF_DIR.glob("*.pdf")):
        if path.name not in known:
            standards.append({"standard_id": "未配置", "title": path.stem, "source_file": path.name, "chunks": 0, "status": "待入库"})
    return {"standards": standards, "vector_model": vector_config.get("model") or vector_config.get("model_name") or "未识别", "total_chunks": len(metas)}


@app.get("/api/policy/version-compare")
def version_compare_topics() -> dict[str, Any]:
    return list_version_topics()


@app.get("/api/policy/version-compare/{topic_id}")
def version_compare_topic(topic_id: str) -> dict[str, Any]:
    try:
        return get_version_topic(topic_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"未找到版本对比主题：{topic_id}") from exc


@app.post("/api/policy/version-compare/resolve")
def resolve_version_question_endpoint(payload: VersionResolveRequest) -> dict[str, Any]:
    try:
        return resolve_version_question(
            payload.question,
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            top_k=payload.top_k,
            applicable_date=payload.applicable_date,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (RuntimeError, FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=502, detail=f"版本对比检索暂时不可用：{exc}") from exc


@app.post("/api/qa")
def ask_qa(payload: QARequest) -> dict[str, Any]:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    state = build_state(payload.history)
    decision = None
    retrieval_question = question
    if payload.history:
        try:
            decision = route_context_with_llm(
                question,
                state,
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                turn_history=compact_turn_history(payload.history),
            )
            decision = normalize_context_decision(question, decision, state=state)
            retrieval_question = decision.resolved_question if decision.use_previous_context else question
        except Exception:
            decision = normalize_context_decision(
                question,
                ContextDecision(
                    mode="new_topic",
                    use_previous_context=False,
                    resolved_question=question,
                    reason="上下文路由模型不可用，已按规则兜底。",
                    source="fallback",
                ),
                state=state,
            )
            retrieval_question = decision.resolved_question if decision.use_previous_context else question

    try:
        candidate_k = max(payload.top_k * 2, 10) if decision and decision.use_previous_context else payload.top_k
        fresh_results = retrieve(
            retrieval_question,
            scope=payload.scope,
            top_k=candidate_k,
            use_reranker=payload.use_reranker,
        )
        context_results = []
        if decision and decision.use_previous_context:
            evidence_sets = 2 if decision.mode in {"comparison", "evidence_request"} else 1
            context_results = recent_evidence(state, sets=evidence_sets, max_items=candidate_k)
        results = select_turn_evidence(
            decision,
            fresh_results,
            context_results,
            top_k=payload.top_k,
            question=question,
        )
        results = annotate_version_applicability(results, payload.applicable_date)
        answer = generate_answer(
            question,
            results,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            retrieval_question=retrieval_question,
            is_followup=bool(decision and decision.use_previous_context),
            is_reference_question=bool(decision and decision.mode in {"evidence_request", "comparison"}),
        )
    except (LLMAnswerError, RuntimeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=502, detail=f"问答服务暂时不可用：{exc}") from exc

    sources = [public_source(item) for item in results[: payload.top_k]]
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "evidence": evidence_assessment(results, answer),
        "context": {
            "used": bool(decision and decision.use_previous_context),
            "mode": decision.mode if decision else "new_topic",
            "reason": decision.reason if decision else "本轮按独立问题检索。",
            "resolved_question": retrieval_question,
        },
    }


@app.post("/api/ifc/parse")
async def parse_ifc(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "model.ifc").suffix.lower()
    if suffix not in {".ifc", ".ifcxml"}:
        raise HTTPException(status_code=400, detail="仅支持 IFC 或 IFCXML 文件")
    safe_name = f"{uuid.uuid4().hex[:12]}_{Path(file.filename or 'model.ifc').name}"
    path = IFC_UPLOAD_DIR / safe_name
    max_bytes = 200 * 1024 * 1024
    total = 0
    try:
        with path.open("wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail="文件不能超过 200MB")
                buffer.write(chunk)
        summary = parse_ifc_file(path)
        return {"summary": summary, "review_questions": build_ifc_review_questions(summary), "processing_chain": ifc_processing_chain()}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"IFC 解析失败：{exc}") from exc
    finally:
        path.unlink(missing_ok=True)


def call_deepseek_json(system_prompt: str, payload: dict[str, Any], timeout: int = 30, max_tokens: int = 700) -> dict[str, Any]:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY 未配置")
    base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    url = os.getenv("DEEPSEEK_CHAT_URL") or f"{base}/chat/completions"
    body = {
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
    last_error: Exception | None = None
    started = time.time()
    usage: dict[str, Any] | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_body = json.loads(response.read().decode("utf-8"))
                usage = response_body.get("usage")
                content = response_body["choices"][0]["message"]["content"]
            break
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {408, 429} and exc.code < 500:
                raise RuntimeError(f"DeepSeek 请求失败：{exc}") from exc
        except urllib.error.URLError as exc:
            last_error = exc
        if attempt < 2:
            time.sleep(0.6 * (attempt + 1))
    else:
        record_llm_usage(None, (time.time() - started) * 1000.0, ok=False)
        raise RuntimeError(f"DeepSeek 请求失败：{last_error}") from last_error
    record_llm_usage(usage, (time.time() - started) * 1000.0)
    clean = content.strip().replace("```json", "").replace("```", "")
    start, end = clean.find("{"), clean.rfind("}")
    return json.loads(clean[start : end + 1])


@app.post("/api/ifc/review-preparation")
def prepare_ifc_review(payload: IFCReviewPreparationRequest) -> dict[str, Any]:
    model_facts = compact_ifc_facts(payload.summary)
    user_profile = normalize_project_profile({}, payload.project_context)
    try:
        raw = call_deepseek_json(
            project_preparation_prompt(),
            {
                "latest_user_project_context": payload.project_context,
                "preclassified_user_conditions": {
                    "confirmed": user_profile["confirmed_conditions"],
                    "uncertain": user_profile["uncertain_conditions"],
                },
                "ifc_facts": model_facts,
            },
            max_tokens=1200,
        )
    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
        fallback = normalize_project_profile(user_profile, payload.project_context)
        fallback["review_questions"] = build_ifc_review_questions(payload.summary)
        return {
            "project_profile": fallback,
            "review_questions": fallback["review_questions"],
            "preparation_status": "fallback",
            "notice": "智能建议暂时不可用，已应用用户项目条件并使用模型字段生成备用复核项。",
        }
    profile = normalize_project_profile(raw, payload.project_context)
    return {
        "project_profile": profile,
        "review_questions": profile["review_questions"],
        "preparation_status": "generated",
        "notice": "",
    }


@app.post("/api/ifc/model-question")
def ask_ifc_model(payload: IFCQuestionRequest) -> dict[str, Any]:
    prompt = (
        "你是 IFC 模型数据分析助手。只能依据提供的 IFC 解析结果回答模型事实问题，不检索建筑规范，"
        "不把模型命名推断当成确定事实。信息不足时明确列出缺失字段。输出 JSON："
        '{"answer":"直接回答","certainty":"high|medium|low","facts_used":["字段和值"],"missing_fields":[],"note":"边界说明"}'
    )
    try:
        result = call_deepseek_json(prompt, {"question": payload.question, "project_context": payload.project_context, "ifc_summary": payload.summary})
    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail=f"模型事实问答暂时不可用：{exc}") from exc
    return result


@app.post("/api/ifc/compliance-question")
def ask_ifc_compliance(payload: IFCQuestionRequest) -> dict[str, Any]:
    summary = payload.summary
    model_facts = compact_ifc_facts(summary)
    supplied_profile = payload.project_profile
    if not supplied_profile and payload.project_context.strip():
        preparation = prepare_ifc_review(
            IFCReviewPreparationRequest(summary=summary, project_context=payload.project_context)
        )
        supplied_profile = preparation.get("project_profile") or {}
    project_profile = normalize_project_profile(supplied_profile, payload.project_context)
    composite_question = compose_compliance_context(
        payload.question,
        payload.project_context,
        model_facts,
        project_profile,
    )
    confirmed_conditions = project_profile.get("confirmed_conditions") or []
    answer_question = payload.question
    if confirmed_conditions:
        answer_question += (
            "\n当前用户已确认的项目条件："
            + "；".join(str(item) for item in confirmed_conditions)
            + "。这些条件是当前审查前提，不得因IFC低置信度推断而再次列为缺失或待确认。"
        )
    try:
        results = retrieve(
            payload.question + " " + payload.project_context,
            scope=payload.scope,
            top_k=payload.top_k,
            use_reranker=payload.use_reranker,
            candidate_top_k=max(payload.top_k, 8),
        )
        results = annotate_version_applicability(results, payload.applicable_date)
        answer = generate_answer(
            answer_question,
            results,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            retrieval_question=composite_question,
            compliance_mode=True,
        )
        answer = apply_project_conflict_guard(answer, project_profile)
    except (LLMAnswerError, RuntimeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=502, detail=f"合规审查暂时不可用：{exc}") from exc
    cited = set(answer.get("citations") or [])
    sources = [
        public_source(item)
        for index, item in enumerate(results, start=1)
        if index in cited or index <= payload.top_k
    ]
    return {
        "question": payload.question,
        "answer": answer,
        "verdict": answer.get("verdict", "INSUFFICIENT_INFORMATION"),
        "sources": sources,
        "evidence": evidence_assessment(results, answer),
        "model_facts": model_facts,
        "project_profile": project_profile,
    }
