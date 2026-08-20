from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

from answer_consistency import apply_answer_consistency_guard
from deterministic_compliance import apply_numeric_compliance_guard
from observability import now_ms, record_llm_usage


DEFAULT_DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"


class LLMAnswerError(RuntimeError):
    pass


TABLE_MAPPING_TERMS = (
    "完整复述",
    "整行",
    "一行",
    "所有",
    "各项",
    "分别",
    "对应",
    "哪一列",
    "表头",
)


def requires_table_mapping(question: str) -> bool:
    """Return whether answering requires a reliable row/column association."""
    compact = re.sub(r"\s+", "", question or "")
    if not compact:
        return False
    explicit_table = "表" in compact and any(term in compact for term in TABLE_MAPPING_TERMS)
    multiple_targets = any(term in compact for term in ("分别", "各自", "两个数值", "这些数值"))
    return explicit_table or multiple_targets


def table_mapping_issues(item: dict) -> list[str]:
    """Detect extraction defects that make table coordinates unsafe to infer."""
    if not (item.get("has_table") or item.get("chunk_type") == "structured_table"):
        return []

    text = str(item.get("text") or "")
    issues: list[str] = []
    grid_match = re.search(r"\[TABLE_START\](.*?)\[TABLE_END\]", text, re.S)
    if not grid_match:
        issues.append("缺少可解析的表格网格")
        return issues

    rows = [line.strip() for line in grid_match.group(1).splitlines() if "|" in line]
    parsed_rows = [[cell.strip() for cell in line.strip(" |").split("|")] for line in rows]
    non_empty_rows = [row for row in parsed_rows if any(row)]
    if len(non_empty_rows) < 2:
        issues.append("表头或数据行不完整")

    widths = {len(row) for row in non_empty_rows}
    if len(widths) > 1:
        issues.append("表格各行列数不一致")

    if any(any(not cell for cell in row) for row in non_empty_rows):
        issues.append("表格存在空单元格")

    # Common OCR damage around numeric cells, independent of any standard or value.
    numeric_damage = re.compile(
        r"(?:[\]\[\"']\s*[.,]?\s*\d)|(?:\d\s*[\"']\s*[.,]?\s*\d)|(?:\d\s*[A-Za-z]\s*(?:m|h|%))"
    )
    if numeric_damage.search(text):
        issues.append("数值单元格存在OCR异常字符")
    return issues


def apply_table_mapping_guard(question: str, candidates: list[dict], answer: dict) -> dict:
    """Prevent confident multi-cell answers when cited table coordinates are unsafe."""
    if not requires_table_mapping(question):
        return answer

    cited = answer.get("citations") or []
    cited_candidates = [candidates[index - 1] for index in cited if 1 <= index <= len(candidates)]
    issue_sets = [table_mapping_issues(item) for item in cited_candidates]
    issues = sorted({issue for item_issues in issue_sets for issue in item_issues})
    if not issues:
        return answer

    missing = answer.setdefault("missing_fields", [])
    field = "表头与数值列的可靠对应关系"
    if field not in missing:
        missing.append(field)
    answer["can_answer"] = False
    answer["certainty"] = "low"
    answer["conclusion"] = "已召回到相关表格，但表格抽取结果不足以可靠确认各字段与数值的对应关系。"
    answer["confidence_reason"] = "当前问题需要逐列映射表格数值，但结构化表格存在：" + "、".join(issues) + "。"
    answer["note"] = "请展开引用来源并核对 PDF 原表；系统不会根据数值顺序猜测列位。"
    return answer


def _chat_url() -> str:
    explicit = os.getenv("DEEPSEEK_CHAT_URL")
    if explicit:
        return explicit
    base = os.getenv("DEEPSEEK_BASE_URL", "").rstrip("/")
    if not base:
        return DEFAULT_DEEPSEEK_URL
    return f"{base}/chat/completions"


def _candidate_context(candidates: list[dict], max_items: int = 8) -> str:
    blocks = []
    for idx, item in enumerate(candidates[:max_items], start=1):
        text = re.sub(r"\s+", " ", item.get("text", "")).strip()
        source_file = item.get("source_file") or item.get("standard_id") or "未知文件"
        page = item.get("page")
        page_text = f"PDF物理页第 {page} 页（含封面、目录）" if page is not None else "PDF物理页未识别"
        table_issues = table_mapping_issues(item)
        table_quality = "、".join(table_issues) if table_issues else "未发现明显结构异常"
        version_note = f"版本适用提示: {item['version_note']}" if item.get("version_note") else "版本适用提示: 未提供项目日期，无法判断版本适用性"
        blocks.append(
            "\n".join(
                [
                    f"[证据{idx}]",
                    f"来源文件: {source_file}",
                    f"标准: {item.get('standard_id')}",
                    f"条文号: {item.get('article_no') or '未识别'}",
                    f"页码: {page_text}",
                    f"表格结构质量: {table_quality}" if item.get("has_table") else "表格结构质量: 非表格证据",
                    version_note,
                    f"原文: {text[:1400]}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def normalize_compliance_answer(answer: dict) -> dict:
    valid_verdicts = {"PASS", "FAIL", "INSUFFICIENT_INFORMATION"}
    verdict = str(answer.get("verdict") or "").strip().upper()
    if verdict not in valid_verdicts:
        verdict = "INSUFFICIENT_INFORMATION"
    reasons = answer.get("compliance_reasons")
    if not isinstance(reasons, list):
        reasons = []
    if not answer.get("can_answer") or answer.get("certainty") == "low" or not reasons:
        verdict = "INSUFFICIENT_INFORMATION"
    answer["verdict"] = verdict
    answer["compliance_reasons"] = [str(item) for item in reasons[:6]]
    answer["critical_risk"] = bool(answer.get("critical_risk")) and verdict == "FAIL"
    conclusion = str(answer.get("conclusion") or "")
    if verdict == "INSUFFICIENT_INFORMATION":
        if "不足" not in conclusion and "无法" not in conclusion and "不能" not in conclusion:
            answer["conclusion"] = "证据不足，无法可靠给出合规结论。"
    elif verdict == "PASS" and re.search(r"不满足|不符合|不通过|超限|(?<![未没])超过|低于要求|小于要求", conclusion):
        answer["conclusion"] = "数值规则复核后，系统提取的实际值满足规范限值；仍需人工核对适用条件。"
    elif verdict == "FAIL" and re.search(r"(?<![不未])满足|(?<![不未])符合|(?<![不未])通过|未超过|没有超过", conclusion):
        answer["conclusion"] = "数值规则复核发现实际值不满足规范限值；请核对原始条文、模型字段和适用条件后确认。"
    return answer


def generate_answer(
    question: str,
    candidates: list[dict],
    *,
    api_key: str | None = None,
    timeout: int = 25,
    retrieval_question: str | None = None,
    is_followup: bool = False,
    is_reference_question: bool = False,
    compliance_mode: bool = False,
) -> dict:
    key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise LLMAnswerError("DEEPSEEK_API_KEY is not set")
    if not candidates:
        return {
            "can_answer": False,
            "conclusion": "未召回到可审查依据，无法生成可靠答案。",
            "basis": "当前候选证据为空。",
            "certainty": "low",
            "missing_fields": ["标准", "条文号", "页码", "原文片段"],
            "citations": [],
            "note": "系统不会根据模型常识补写答案。",
            "verdict": "INSUFFICIENT_INFORMATION",
            "compliance_reasons": ["未召回到可审查依据。"],
            "critical_risk": False,
        }

    system_prompt = """你是建筑消防规范审查助手。你只能依据用户提供的证据片段回答，不能使用预训练记忆补充规范条文、数值、页码或标准号。

规则：
1. 如果证据原文明确包含可支持结论的标准、条文号、页码和关键数值/要求，can_answer 才能为 true，certainty 才能为 high。
2. 如果证据相关但没有直接给出关键数值、条文号或完整条件，certainty 为 medium，missing_fields 必须列出缺哪个字段。
3. 如果证据不相关或不足以回答，can_answer 必须为 false，certainty 为 low，conclusion 必须写“未召回到足够依据，无法可靠回答”。
4. basis 必须写出采用的来源文件、标准、条文号、PDF物理页和原文关键词。不得编造不存在的条文号、页码或原文。
5. citations 只能填写证据编号，例如 [1, 3]；不得引用未提供的证据编号。
6. confidence_reason 必须解释 certainty 的来源，尤其是 medium / low 时要说明缺什么、为什么需要复核。
7. 适用对象、限定条件、例外条件和时间状态都是结论的一部分。证据只支持某个子场景时，不得把它推广到更宽泛的上位概念。
8. 当证据按“新建/既有改造”“设置/未设置”“地上/地下”“直通室外/其他位置”等条件给出不同要求时，用户未明确条件就必须分条件回答，不能擅自选择其中一个数值。
9. 如果用户问题范围比知识库证据更宽，必须明确当前知识库能回答的范围和仍缺少的规范或条件；不得用模型记忆补齐。
10. 问题要求把多个数值分别对应到表格字段时，必须检查“表格结构质量”。若存在列数不一致、空单元格、数据行不完整或数字 OCR 异常，不得根据数值顺序猜测列位，can_answer 必须为 false。
11. atomic_claims 必须把 conclusion 中可独立核验的规范要求、项目事实和计算结果逐项拆开；每项只表达一个判断，并分别绑定证据编号。项目事实和计算结果可引用用户输入，其 kind 分别为 project_fact、calculation；规范要求 kind 为 requirement。
12. 输出必须是 JSON，不要输出 Markdown。

合规审查模式：
1. 只有当证据同时覆盖模型事实、适用条件和规范条文时，verdict 才能为 PASS 或 FAIL。
2. 缺少建筑用途、构件用途、空间关系、规范适用版本、关键数值或完整条件时，verdict 必须为 INSUFFICIENT_INFORMATION。
3. 实际不满足规范要求时必须使用 FAIL，不得为了“回答得更客气”改成 PASS。
4. compliance_reasons 必须逐条说明支撑结论的 IFC 事实、规范依据和计算/判断过程。
5. critical_risk 只在 verdict 为 FAIL 且存在安全出口、疏散距离、耐火性能等安全关键风险时为 true。
6. conclusion 必须与 verdict 保持一致：PASS 时明确满足，FAIL 时明确不满足，INSUFFICIENT_INFORMATION 时明确证据不足。

JSON 格式：
{
  "can_answer": true,
  "conclusion": "一句话直接结论；证据不足时明确说不能回答",
  "basis": "来源文件 + 标准 + 条文号 + PDF物理页 + 原文关键词/数值；缺失字段要明说",
  "certainty": "high / medium / low",
  "confidence_reason": "为什么给出这个模型判断置信度",
  "missing_fields": [],
  "citations": [1],
  "note": "人工复核提醒或不确定原因",
  "verdict": "PASS / FAIL / INSUFFICIENT_INFORMATION",
  "compliance_reasons": ["模型事实", "规范依据", "计算或判断"],
  "critical_risk": false,
  "atomic_claims": [
    {"text": "一个可独立核验的判断", "kind": "requirement / project_fact / calculation", "citations": [1]}
  ]
}
"""
    answer_shape = (
        "回答组织要求："
        "1. conclusion 必须先直接回答用户本轮问题，优先写成 1-2 句短答。"
        "2. 如果这是追问，只回答本轮新增条件、差异点或用户追问的对象；不要完整复述上一轮已经回答过的全量规则，除非不复述会导致结论不完整。"
        "3. 如果用户问“为什么不同/是否同一条/来自哪条原文/列出表号页码”，必须围绕上一轮证据做解释或列举，不要重新回答无关条文。"
        "4. 如果问题涉及数值，必须写清楚该数值适用于哪类建筑、哪个部位、什么条件。"
        "5. basis 只写支撑本轮结论的关键来源，格式为“来源文件；标准；第 X 条；PDF物理页第 Y 页；原文关键词”。不要把所有候选证据都堆进去。"
        "6. 不得把旧规范条文直接当作最终结论；如同时召回现行通用规范，应说明通用规范与专项规范的关系。"
        "7. 生成结论前逐项核对数值前的主语、建筑状态、部位和例外条件；同一对象存在多个条件分支时必须完整列出。"
    )
    if compliance_mode:
        answer_shape += (
            "\n合规数值校验：\n"
            "8. 当结论依赖可计算的实际值与规范限值时，必须输出 numeric_checks。\n"
            "9. numeric_checks 每项包含 citation、metric、actual_value、actual_unit、"
            "requirement_value、requirement_unit、comparator、expected_result。\n"
            "10. comparator 表示实际值满足规范时应满足的方向，只能使用 lte、lt、gte、gt、eq。"
            "例如最大面积用 lte，最小净宽用 gte。\n"
            "11. expected_result 是本条数值规则自身的判定，只能是 PASS 或 FAIL；不得把需要人工确认的字段填入数值。\n"
            "12. 同时输出 decision_basis：仅由这些数值规则决定时写 numeric_only；"
            "还存在用途、部位、规范版本等非数值条件时写 mixed。\n"
            "13. 输出 blocking_conditions：只列出会改变适用条款、限值或最终PASS/FAIL的未确认项目条件；没有则为空数组。\n"
            "14. 输出 advisory_risks：列出不改变当前原子结论、但仍建议人工复核的OCR、来源完整性或数据质量提示；这些提示本身不阻断PASS/FAIL。\n"
            "15. non_numeric_failures 作为兼容字段，内容必须与 blocking_conditions 相同；不得把纯OCR提示或已被证据审计排除的补充结论写入其中。\n"
        )
    dialogue_mode = []
    if is_followup:
        dialogue_mode.append("这是连续追问，请承接上下文并保持短答。")
    if is_reference_question:
        dialogue_mode.append("这是证据追溯/对比问题，请优先解释上一轮证据和条文来源。")
    retrieval_line = ""
    if retrieval_question and retrieval_question != question:
        retrieval_line = f"\n检索补全文（只用于理解上下文，不要照抄到回答中）：{retrieval_question}\n"
    user_prompt = (
        f"用户本轮问题：{question}"
        f"{retrieval_line}\n"
        f"{' '.join(dialogue_mode)}\n\n"
        f"{answer_shape}\n\n召回证据：\n{_candidate_context(candidates)}"
    )
    payload = {
        "model": os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 1200,
        "response_format": {"type": "json_object"},
    }
    request_data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    body = ""
    last_error: Exception | None = None
    started = now_ms()
    for attempt in range(3):
        request = urllib.request.Request(
            _chat_url(),
            data=request_data,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
            last_error = None
            break
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            last_error = LLMAnswerError(f"LLM request failed: HTTP {exc.code} {error_body[:300]}")
            if exc.code not in {408, 429, 500, 502, 503, 504}:
                raise last_error from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            last_error = LLMAnswerError(f"LLM request failed: {reason}")
        if attempt < 2:
            time.sleep(1.0 * (attempt + 1))
    if last_error is not None:
        record_llm_usage(None, now_ms() - started, ok=False)
        raise last_error

    data = json.loads(body)
    record_llm_usage(data.get("usage"), now_ms() - started)
    content = data["choices"][0]["message"]["content"]
    parsed = _extract_json(content)
    certainty = parsed.get("certainty", "low")
    if certainty not in {"high", "medium", "low"}:
        parsed["certainty"] = "low"
        parsed["note"] = "模型返回的置信度字段异常，已降级为低置信度。"
    citations = parsed.get("citations")
    if not isinstance(citations, list):
        parsed["citations"] = []
    else:
        parsed["citations"] = [int(x) for x in citations if isinstance(x, int) and 1 <= x <= len(candidates)]
    missing_fields = parsed.get("missing_fields")
    if not isinstance(missing_fields, list):
        parsed["missing_fields"] = []
    confidence_reason = parsed.get("confidence_reason")
    if not isinstance(confidence_reason, str) or not confidence_reason.strip():
        if parsed["certainty"] == "high":
            parsed["confidence_reason"] = "证据直接包含可支持结论的原文、条文号、页码和关键字段。"
        elif parsed["certainty"] == "medium":
            missing = "、".join([str(item) for item in parsed["missing_fields"]]) or "部分适用条件"
            parsed["confidence_reason"] = f"证据相关但仍需复核：{missing}。"
        else:
            parsed["confidence_reason"] = str(parsed.get("note") or "证据不足或不相关，不能可靠支持结论。")
    parsed["can_answer"] = bool(parsed.get("can_answer")) and parsed["certainty"] in {"high", "medium"}
    if parsed["can_answer"] and not parsed["citations"]:
        parsed["certainty"] = "low"
        parsed["can_answer"] = False
        parsed["note"] = "模型未返回可追溯证据编号，已降级为低置信度。"
    if parsed["certainty"] == "low":
        parsed["can_answer"] = False
    result = apply_table_mapping_guard(question, candidates, parsed)
    grounding_text = "\n".join(part for part in (question, retrieval_question or "") if part)
    result = apply_answer_consistency_guard(
        result,
        candidates,
        grounding_text=grounding_text,
        allow_partial_claims=compliance_mode,
    )
    if compliance_mode:
        result = apply_numeric_compliance_guard(
            result,
            retrieval_question=retrieval_question or "",
            candidates=candidates,
        )
        result = normalize_compliance_answer(result)
    return result
