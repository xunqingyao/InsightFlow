"""研究模型适配层。

两种实现：
- `OfflineResearchModel`：完全离线、确定性的模板实现，用于回放演示与测试。
  它只从已抓取的正文里挑句子，不生成任何没有来源的事实。
- `OpenAIResearchModel`：OpenAI 兼容的 `/chat/completions` 接口，要求模型返回
  严格 JSON；返回的每条 findings 都必须携带 source_id + quote，quote 会被校验
  必须是原文子串，否则该条被丢弃。

两者的输出都先落到 `Synthesis` 结构，再由编排层登记 evidence、渲染报告。
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from insightflow.config.settings import Settings
from insightflow.research.schemas import (
    ClaimType,
    ModelUsage,
    ResearchRequest,
    Source,
    SupportStatus,
)
from insightflow.research.time_window import describe_window


class LLMError(Exception):
    """模型不可用或输出无法解析。"""


class SynthesisEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    quote: str = Field(min_length=1, max_length=2000)
    locator: str = "正文"


class SynthesisFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=4, max_length=1000)
    claim_type: ClaimType = ClaimType.PUBLISHER_CLAIM
    support_status: SupportStatus = SupportStatus.PARTIAL
    evidence: list[SynthesisEvidence] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class Synthesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = ""
    findings: list[SynthesisFinding] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class SourceMaterial(BaseModel):
    """交给模型的单个来源材料。`text` 已按字符上限截断。"""

    source_id: str
    title: str
    url: str
    kind: str
    role: str
    published_at: datetime | None
    text: str


class SourceBrief(BaseModel):
    """决定下一步时看到的来源清单——不包含正文，避免把决策变成又一次全文阅读。"""

    source_id: str
    title: str
    kind: str
    role: str
    published_at: datetime | None
    fetch_scope: str


class GapBrief(BaseModel):
    kind: str
    theme: str | None = None
    label: str = ""
    reason: str
    queries: list[str] = Field(default_factory=list)


class ResearchState(BaseModel):
    """模型决定“继续查还是收工”时能看到的全部事实。"""

    request: ResearchRequest
    sources: list[SourceBrief] = Field(default_factory=list)
    gaps: list[GapBrief] = Field(default_factory=list)
    findings: int = 0
    rounds_used: int = 0
    rounds_left: int = 0
    searches_left: int = 0


class Decision(BaseModel):
    """模型的下一步决定：补搜（给出检索词）或停止（给出理由）。"""

    action: Literal["supplement", "stop"]
    reason: str = ""
    queries: list[str] = Field(default_factory=list)


class ResearchModel(Protocol):
    name: str
    provider: str
    usage: ModelUsage | None

    def plan(self, request: ResearchRequest) -> list[str]: ...

    def synthesize(
        self, request: ResearchRequest, materials: list[SourceMaterial]
    ) -> Synthesis: ...

    def decide(self, state: ResearchState) -> Decision: ...


# ---------------------------------------------------------------- offline


_SENTENCE = re.compile(r"(?<=[。！？!?])\s*|\n+")

# 样板句/免责声明不承载信息，抽取时跳过，避免报告里塞满"本文为示例"。
_BOILERPLATE = ("虚构示例", "请勿引用", "不对应真实", "不构成独立验证", "示例数据", "本文为")


def _sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE.split(text) if part and part.strip()]
    kept = [
        part
        for part in parts
        if len(part) >= 20 and not any(marker in part for marker in _BOILERPLATE)
    ]
    return kept


class OfflineResearchModel:
    """模板模型：只做抽取与组织，不做事实推断。"""

    name = "offline-template"
    provider = "offline"

    def __init__(self) -> None:
        self.usage: ModelUsage | None = None

    def plan(self, request: ResearchRequest) -> list[str]:
        base = " ".join(request.query.split())
        queries = [f"{base} 发布", f"{base} 更新"]
        for topic in request.focus[:2]:
            queries.append(f"{base} {topic}")
        return queries[:4]

    def synthesize(
        self, request: ResearchRequest, materials: list[SourceMaterial]
    ) -> Synthesis:
        findings: list[SynthesisFinding] = []
        for material in materials:
            quote = self._pick_quote(material.text)
            if not quote:
                continue
            when = (
                material.published_at.astimezone(UTC).strftime("%Y-%m-%d")
                if material.published_at
                else "发布时间未知"
            )
            short_title = material.title if len(material.title) <= 40 else material.title[:38] + "…"
            findings.append(
                SynthesisFinding(
                    text=f"{short_title}（{when}）：{quote}",
                    claim_type=ClaimType.PUBLISHER_CLAIM,
                    support_status=SupportStatus.PARTIAL,
                    evidence=[
                        SynthesisEvidence(
                            source_id=material.source_id, quote=quote, locator="正文"
                        )
                    ],
                    limitations=["单一来源陈述，未见独立第三方交叉验证"],
                )
            )
        dated = sum(1 for item in materials if item.published_at is not None)
        summary = (
            f"本报告基于 {len(materials)} 个来源，其中 {dated} 个可确认发布时间；"
            f"时间窗口为 {describe_window(request.window_start, request.window_end)}"
            f"（as_of {request.as_of.isoformat()}）。"
            "内容为来源陈述的摘录整理，未做超出证据的推断。"
        )
        limitations = [
            "模板化生成：只摘录来源原句，未进行跨来源综合判断。",
            "未接入实时检索时，结论覆盖范围仅限于已抓取来源。",
        ]
        if dated < len(materials):
            limitations.append(f"{len(materials) - dated} 个来源缺少可确认的发布时间。")
        return Synthesis(summary=summary, findings=findings, limitations=limitations)

    def decide(self, state: ResearchState) -> Decision:
        """模板决策：只看缺口清单，够不够、要不要补都按规则说清楚。"""
        if not state.gaps:
            return Decision(action="stop", reason="覆盖检查没有发现缺口")
        if state.rounds_left <= 0:
            return Decision(action="stop", reason="补搜轮次已用完")
        if state.searches_left <= 0:
            return Decision(action="stop", reason="检索预算已用完")
        queries: list[str] = []
        for gap in state.gaps:
            for query in gap.queries:
                if query and query not in queries:
                    queries.append(query)
        if not queries:
            return Decision(action="stop", reason="缺口没有给出可用的检索词")
        return Decision(
            action="supplement",
            reason="；".join(gap.reason for gap in state.gaps),
            queries=queries[:3],
        )

    @staticmethod
    def _pick_quote(text: str, max_len: int = 240) -> str:
        candidates = [s for s in _sentences(text) if 30 <= len(s) <= max_len]
        if not candidates:
            return ""
        # 优先含数字/技术细节的句子，其次取最长句
        def score(sentence: str) -> tuple[int, int]:
            informative = 1 if re.search(r"\d", sentence) else 0
            return (informative, len(sentence))

        return max(candidates, key=score)


# ------------------------------------------------------------ openai-compatible


_SYSTEM_PROMPT = """你是严谨的技术情报研究员。你只能使用给定的来源材料，不得引入外部知识。
输出必须是单个 JSON 对象，不要 Markdown 代码块，字段如下：
{
  "summary": "200 字以内的范围说明与总体判断",
  "findings": [
    {
      "text": "一条具体结论，写清主体与动作",
      "claim_type": "observable_fact|publisher_claim|research_result|external_test|system_inference",
      "support_status": "supported|partial|conflicted|insufficient",
      "evidence": [{"source_id": "给定 ID", "quote": "来源原文中逐字出现的句子", "locator": "正文/摘要"}],
      "limitations": ["这条结论的边界"]
    }
  ],
  "limitations": ["整体局限"]
}
规则：
1. quote 必须从来源正文逐字复制，长度 20-300 字；不允许改写或拼接。
2. 只有官方公告/文档/仓库等一手来源且内容明确时才能用 observable_fact；媒体报道用 publisher_claim。
3. 转载、二手报道不能作为 supported 的唯一依据。
4. 没有证据支撑的判断写成 system_inference，并在 limitations 里说明依据。
5. 3-6 条 findings，宁少勿多。"""


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        raise LLMError("模型输出中没有 JSON 对象")
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"模型输出不是合法 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise LLMError("模型输出的 JSON 顶层必须是对象")
    return payload



_DECISION_PROMPT = """你是研究流程的调度者，不是写报告的人。你看到：已读来源清单、
覆盖缺口、已用轮次与剩余预算。请决定“继续补搜”还是“停止”，并说明理由。

只输出 JSON：
{"action": "supplement" | "stop", "reason": "一句话理由", "queries": ["检索式"]}

判断规则：
- 缺口为空就停止，不要为了凑数量继续搜。
- 有缺口、且剩余轮次与检索次数都大于 0 时，才允许补搜，并给出针对缺口的检索式（1-3 条）。
- 补搜检索式要具体、彼此不重复，不要原样重复已用过的检索式。
- 预算或轮次不足时停止，并在 reason 里说明是哪一项不足。
"""

class OpenAIResearchModel:
    name = "openai-compatible"
    provider = "openai-compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 90.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client = client
        self.usage: ModelUsage | None = None
        self.calls = 0

    def _post(self, messages: list[dict[str, str]], *, temperature: float = 0.2) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {"authorization": f"Bearer {self.api_key}", "content-type": "application/json"}
        own = self._client is None
        client = self._client or httpx.Client(timeout=self.timeout)
        try:
            response = client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
        except httpx.TimeoutException as exc:
            raise LLMError(f"模型请求超时：{exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"模型网络错误：{exc}") from exc
        finally:
            if own:
                client.close()
        self.calls += 1
        if response.status_code >= 400:
            raise LLMError(f"模型返回 HTTP {response.status_code}：{response.text[:300]}")
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError("模型响应不是 JSON") from exc
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("模型响应缺少 choices[0].message.content") from exc
        usage = data.get("usage") or {}
        self.usage = ModelUsage(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            estimated=False,
            model=self.model,
            provider="openai-compatible",
        )
        return str(content)

    def plan(self, request: ResearchRequest) -> list[str]:
        content = self._post(
            [
                {"role": "system", "content": "只输出 JSON：{\"queries\": [\"...\"]}，3-5 条检索式。"},
                {
                    "role": "user",
                    "content": (
                        f"研究问题：{request.query}\n"
                        f"任务类型：{request.task_type}\n"
                        f"时间窗口：{describe_window(request.window_start, request.window_end)}\n"
                        f"关注点：{', '.join(request.focus) or '无'}\n"
                        "检索式要具体、彼此不重复，便于找到一手来源。"
                    ),
                },
            ]
        )
        payload = _extract_json(content)
        queries = payload.get("queries")
        if not isinstance(queries, list):
            raise LLMError("plan 输出缺少 queries 列表")
        cleaned = [str(item).strip() for item in queries if str(item).strip()]
        if not cleaned:
            raise LLMError("plan 输出为空")
        return cleaned[:5]

    def synthesize(
        self, request: ResearchRequest, materials: list[SourceMaterial]
    ) -> Synthesis:
        blocks: list[str] = []
        for index, material in enumerate(materials, start=1):
            when = material.published_at.isoformat() if material.published_at else "未知"
            blocks.append(
                f"[{index}] source_id={material.source_id}\n"
                f"标题：{material.title}\n"
                f"URL：{material.url}\n"
                f"类型：{material.kind}／角色：{material.role}／发布时间：{when}\n"
                f"正文摘录：\n{material.text}\n"
            )
        user = (
            f"研究问题：{request.query}\n"
            f"任务类型：{request.task_type}\n"
            f"时间窗口：{describe_window(request.window_start, request.window_end)}"
            f"（左闭右开，as_of={request.as_of.isoformat()}）\n"
            f"关注点：{', '.join(request.focus) or '无'}\n\n"
            "以下是本次检索到的来源材料：\n\n" + "\n".join(blocks)
        )
        last_error: Exception | None = None
        for attempt in range(2):
            content = self._post(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=0.2 if attempt == 0 else 0.0,
            )
            try:
                return Synthesis.model_validate(_extract_json(content))
            except (LLMError, ValidationError) as exc:
                last_error = exc
        raise LLMError(f"模型输出无法解析为 Synthesis：{last_error}")



    def decide(self, state: ResearchState) -> Decision:
        payload = {
            "question": state.request.query,
            "window": describe_window(state.request.window_start, state.request.window_end),
            "focus": state.request.focus,
            "sources": [
                {
                    "source_id": item.source_id,
                    "title": item.title,
                    "kind": item.kind,
                    "role": item.role,
                    "published_at": item.published_at.isoformat() if item.published_at else None,
                }
                for item in state.sources
            ],
            "gaps": [gap.model_dump(mode="json") for gap in state.gaps],
            "findings": state.findings,
            "rounds_used": state.rounds_used,
            "rounds_left": state.rounds_left,
            "searches_left": state.searches_left,
        }
        content = self._post(
            [
                {"role": "system", "content": _DECISION_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        )
        try:
            data = _extract_json(content)
        except LLMError:
            raise
        action = str(data.get("action") or "").strip().lower()
        reason = str(data.get("reason") or "").strip()
        raw_queries = data.get("queries") or []
        if not isinstance(raw_queries, list):
            raise LLMError("decide 输出里 queries 不是列表")
        queries: list[str] = []
        for item in raw_queries:
            text = str(item).strip()
            if text and text not in queries:
                queries.append(text)
        if action not in {"supplement", "stop"}:
            raise LLMError(f"decide 输出 action 非法：{action!r}")
        if action == "supplement" and not queries:
            raise LLMError("decide 要求补搜，但没有给出检索式")
        return Decision(action=action, reason=reason, queries=queries[:3])


def build_model(settings: Settings, *, prefer_offline: bool) -> ResearchModel:
    if prefer_offline or not settings.model_configured:
        return OfflineResearchModel()
    assert settings.model_base_url and settings.model_api_key and settings.model_name
    return OpenAIResearchModel(
        base_url=settings.model_base_url,
        api_key=settings.model_api_key,
        model=settings.model_name,
        timeout=settings.model_timeout_seconds,
    )


__all__ = [
    "Decision",
    "GapBrief",
    "LLMError",
    "OfflineResearchModel",
    "OpenAIResearchModel",
    "ResearchModel",
    "ResearchState",
    "SourceBrief",
    "SourceMaterial",
    "Synthesis",
    "SynthesisEvidence",
    "SynthesisFinding",
    "Literal",
    "build_model",
]