from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field


DEFAULT_DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"


@dataclass
class ContextDecision:
    mode: str = "new_topic"
    use_previous_context: bool = False
    resolved_question: str = ""
    reason: str = ""
    context_used: list[str] = field(default_factory=list)
    source: str = "fallback"


def _chat_url() -> str:
    explicit = os.getenv("DEEPSEEK_CHAT_URL")
    if explicit:
        return explicit
    base = os.getenv("DEEPSEEK_BASE_URL", "").rstrip("/")
    if not base:
        return DEFAULT_DEEPSEEK_URL
    return f"{base}/chat/completions"


def _extract_json(text: str) -> dict:
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _compact_history(state, max_items: int = 4, turn_history: list[dict] | None = None) -> list[dict]:
    if turn_history:
        rows: list[dict] = []
        for idx, turn in enumerate(turn_history[-max_items:], start=max(1, len(turn_history) - max_items + 1)):
            row = {"turn": idx, "question": str(turn.get("question") or "")}
            evidence = []
            for item in (turn.get("sources") or [])[:2]:
                evidence.append(
                    {
                        "standard_id": item.get("standard_id"),
                        "article_no": item.get("article_no"),
                        "page": item.get("page"),
                    }
                )
            if evidence:
                row["evidence"] = evidence
            rows.append(row)
        return rows

    questions = list(getattr(state, "context_questions", []) or [])[-max_items:]
    evidence_sets = list(getattr(state, "evidence_sets", []) or [])[-max_items:]
    rows: list[dict] = []
    for idx, question in enumerate(questions, start=max(1, len(questions) - max_items + 1)):
        rows.append({"turn": idx, "question": question})
    if evidence_sets:
        latest = evidence_sets[-1]
        evidence = []
        for item in latest[:3]:
            evidence.append(
                {
                    "standard_id": item.get("standard_id"),
                    "article_no": item.get("article_no"),
                    "page": item.get("page"),
                }
            )
        if evidence:
            rows.append({"latest_evidence": evidence})
    return rows


def route_context_with_llm(
    question: str,
    state,
    *,
    api_key: str | None = None,
    timeout: int = 12,
    turn_history: list[dict] | None = None,
) -> ContextDecision:
    key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    history = _compact_history(state, turn_history=turn_history)
    if not history:
        return ContextDecision(
            mode="new_topic",
            use_previous_context=False,
            resolved_question=question,
            reason="没有历史上下文，本轮按独立问题处理。",
            source="llm",
        )

    system_prompt = """你是 RAG 对话上下文路由器，只判断用户本轮问题是否需要继承上一轮上下文，不回答规范问题。

判定原则：
1. 只有当本轮问题缺少独立对象，或明显使用“这个/那/上述/刚才/这两个/对应条文/哪条原文”等指代时，才 use_previous_context=true。
2. 如果本轮已经包含明确对象和检查目标，即使出现“可以吗/是否/如果”，也应按 new_topic 处理。
3. 如果本轮对象或检查项与上一轮明显不同，应按 new_topic 处理。
4. resolved_question 只能补全必要上下文，不能加入答案、数值、条文号或未在历史/问题中出现的信息。
5. resolved_question 要改写成一个独立、自然、可直接检索的问题，不要简单拼接多轮原句。
6. “那某部位呢”表示只保留用户此前明确给出的建筑场景和项目条件，并替换检查项；不得继承上一轮回答自行采用的计算口径、公式、单位或附加条件。
7. “这两个/分别”要展开为历史问题中真实出现的两个对象或属性；证据追溯才保留上一轮条文定位。
8. 历史回答和证据可能是错误的，只能用于解析指代，不能把其中的数值、计算口径或结论直接写入 resolved_question。
9. 如果本轮出现“回到/返回/刚才/这个结论/来自哪一条/为什么这两个数值不一样/上一轮”等指代历史或对比信号，即使包含明确对象，也应优先继承上下文，并把模式识别为 evidence_request 或 comparison。
10. evidence_request / comparison 的 resolved_question 应尽量还原被追溯或对比的历史问题本身，不要改写成另一个规范问题。
11. 输出必须是 JSON，不要 Markdown。

JSON:
{
  "mode": "new_topic / follow_up / evidence_request / comparison / correction",
  "use_previous_context": false,
  "resolved_question": "用于检索的改写问题",
  "reason": "一句话解释为什么继承或不继承",
  "context_used": ["上一轮问题", "上一轮证据"]
}
"""
    user_prompt = json.dumps(
        {
            "current_question": question,
            "recent_history": history,
        },
        ensure_ascii=False,
        indent=2,
    )
    payload = {
        "model": os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 360,
    }
    request = urllib.request.Request(
        _chat_url(),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Context router failed: HTTP {exc.code} {detail[:200]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Context router failed: {exc.reason}") from exc

    content = json.loads(body)["choices"][0]["message"]["content"]
    parsed = _extract_json(content)
    mode = str(parsed.get("mode") or "new_topic")
    use_previous = bool(parsed.get("use_previous_context")) and mode != "new_topic"
    resolved = str(parsed.get("resolved_question") or question).strip() or question
    context_used = parsed.get("context_used")
    if not isinstance(context_used, list):
        context_used = []
    return ContextDecision(
        mode=mode,
        use_previous_context=use_previous,
        resolved_question=resolved,
        reason=str(parsed.get("reason") or ""),
        context_used=[str(item) for item in context_used],
        source="llm",
    )
