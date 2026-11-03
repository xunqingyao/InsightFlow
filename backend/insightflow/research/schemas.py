"""核心数据结构。

对应设计文档 §7.1。约束刻意收紧：未知时间就是 None，不允许用"现在"兜底；
阅读深度与获取范围分开记录，避免把"下载了全文"当成"模型读了全文"。
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskType(StrEnum):
    BRIEFING = "briefing"
    TOPIC_RESEARCH = "topic_research"
    COMPARISON = "comparison"
    DATA_ANALYSIS = "data_analysis"


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_INPUT = "waiting_input"
    RECOVERABLE = "recoverable"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SourceKind(StrEnum):
    OFFICIAL = "official"
    DOCS = "docs"
    PAPER = "paper"
    REPO = "repo"
    MEDIA = "media"
    COMMUNITY = "community"
    PUBLIC_DATA = "public_data"


class SourceRole(StrEnum):
    """来源在事件中的角色。转载不等于独立验证。"""

    ORIGINAL = "original"
    OFFICIAL = "official"
    REPRINT = "reprint"
    SECONDARY = "secondary"
    UNKNOWN = "unknown"


class ReadDepth(StrEnum):
    """实际提供给研究模型的阅读范围。"""

    NONE = "none"
    SNIPPET = "snippet"
    PARTIAL = "partial"
    FULL = "full"


class FetchScope(StrEnum):
    """实际抓取到的范围。"""

    NONE = "none"
    SNIPPET = "snippet"
    PARTIAL = "partial"
    FULL = "full"


class ClaimType(StrEnum):
    OBSERVABLE_FACT = "observable_fact"
    PUBLISHER_CLAIM = "publisher_claim"
    RESEARCH_RESULT = "research_result"
    EXTERNAL_TEST = "external_test"
    SYSTEM_INFERENCE = "system_inference"


class SupportStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    CONFLICTED = "conflicted"
    INSUFFICIENT = "insufficient"


class SourceStatus(StrEnum):
    FETCHED = "fetched"
    FAILED = "failed"
    BLOCKED = "blocked"


class RunMode(StrEnum):
    ONLINE = "online"
    REPLAY = "replay"


def stable_id(prefix: str, *parts: str) -> str:
    """可复算的短 ID：同一输入得到同一 ID，便于测试与去重。"""
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    task_type: TaskType
    as_of: datetime
    timezone: str
    window_start: datetime
    window_end: datetime
    focus: list[str] = Field(default_factory=list)
    language: str = "zh-CN"
    output_format: str = "markdown"
    budget_profile: str = "standard"
    mode: RunMode = RunMode.ONLINE
    fixture_id: str | None = None

    @model_validator(mode="after")
    def _check_window(self) -> ResearchRequest:
        if self.as_of.tzinfo is None:
            raise ValueError("as_of 必须是带时区的时间")
        if self.window_start.tzinfo is None or self.window_end.tzinfo is None:
            raise ValueError("时间窗口必须带时区")
        if self.window_start >= self.window_end:
            raise ValueError("window_start 必须早于 window_end")
        # 窗口以 as_of 为右端点（左闭右开）
        if self.window_end != self.as_of:
            raise ValueError("window_end 必须等于 as_of")
        if self.mode is RunMode.REPLAY and not self.fixture_id:
            raise ValueError("回放模式必须指定 fixture_id")
        return self


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    url: str
    canonical_url: str
    title: str
    kind: SourceKind = SourceKind.MEDIA
    role: SourceRole = SourceRole.UNKNOWN
    published_at: datetime | None = None
    updated_at: datetime | None = None
    event_at: datetime | None = None
    retrieved_at: datetime
    snapshot_path: str | None = None
    content_hash: str | None = None
    fetch_scope: FetchScope = FetchScope.NONE
    read_depth: ReadDepth = ReadDepth.NONE
    status: SourceStatus = SourceStatus.FETCHED
    error_code: str | None = None
    provider: str = "unknown"

    @property
    def display_time(self) -> datetime | None:
        return self.published_at or self.event_at


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    source_id: str
    locator: str
    quote: str = Field(min_length=1, max_length=2000)
    scope: str | None = None
    created_at: datetime


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str
    text: str = Field(min_length=1, max_length=4000)
    claim_type: ClaimType
    evidence_ids: Annotated[list[str], Field(min_length=0)]
    support_status: SupportStatus
    limitations: list[str] = Field(default_factory=list)


class CheckStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class CheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check: str
    status: CheckStatus
    detail: str


class Report(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_id: str
    run_id: str
    version: int = 1
    created_at: datetime
    title: str
    scope_text: str
    summary: str
    findings: list[Finding]
    limitations: list[str] = Field(default_factory=list)
    cited_source_ids: list[str] = Field(default_factory=list)
    markdown: str = ""
    checks: list[CheckResult] = Field(default_factory=list)
    coverage: list[str] = Field(default_factory=list)
    supplement_rounds: int = 0
    status: Literal["draft", "published", "partial"] = "draft"


class RunCounters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    search_calls: int = 0
    source_fetches: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    supplement_rounds: int = 0
    review_revisions: int = 0
    active_seconds: int = 0
    resumes: int = 0


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: RunStatus
    request: ResearchRequest
    created_at: datetime
    updated_at: datetime
    report_id: str | None = None
    error: str | None = None
    stop_reason: str | None = None
    counters: RunCounters = Field(default_factory=RunCounters)


class ModelUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated: bool = False
    model: str
    provider: str