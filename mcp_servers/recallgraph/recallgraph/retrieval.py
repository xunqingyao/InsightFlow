"""检索契约：一次查询的输入、输出、以及可解释的中间轨迹。

对应文档：docs/02-检索契约.md。

和普通 RAG 最大的区别：**检索单位是断言（Claim），不是段落（Chunk）**。
段落只是证据的载体，最终给 Agent 的是"哪条结论 + 它出自哪一页哪一句"。

    result = retrieve(RetrievalQuery(text="RAG 比微调好吗"))
    for hit in result.claims:
        print(hit.claim.claim_text, hit.citations[0].page)
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from recallgraph.schema import Claim, Note
from recallgraph.vocabulary import (
    ClaimStatus,
    Confidence,
    EntityKind,
    NoteRelation,
    Predicate,
)

RetrievalMode = Literal["evidence", "graph", "hybrid"]
Channel = Literal["lexical", "dense", "graph", "rerank"]
ResultStatus = Literal["ok", "empty", "ambiguous"]
TraceStage = Literal[
    "lexical",
    "dense",
    "fuse",
    "map_entities",
    "expand",
    "assemble",
    "rerank",
    "budget",
]

# 允许用于图扩展的谓词。proposes / evaluates_on 是论文元信息，不用于扩展；
# contradicts 单独走 conflicts 通道，避免"因为存在冲突就把冲突内容当证据"。
EXPANSION_PREDICATES: frozenset[Predicate] = frozenset(
    {
        Predicate.IMPROVES,
        Predicate.OUTPERFORMS,
        Predicate.REQUIRES,
        Predicate.USES,
        Predicate.IS_A,
        Predicate.PART_OF,
        Predicate.CAUSES,
        Predicate.REPLACES,
    }
)


class QueryFilters(BaseModel):
    """结构化过滤。宁可多一个字段，也不要让调用方在结果里自己筛。"""

    model_config = ConfigDict(extra="forbid")

    document_ids: tuple[str, ...] = ()
    version_ids: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()
    entity_kinds: tuple[EntityKind, ...] = ()
    year_min: int | None = Field(default=None, ge=1900, le=2100)
    year_max: int | None = Field(default=None, ge=1900, le=2100)
    languages: tuple[str, ...] = ()
    include_notes: bool = True
    claim_status: tuple[ClaimStatus, ...] = (ClaimStatus.ACTIVE,)

    @model_validator(mode="after")
    def _check_range(self) -> QueryFilters:
        if self.year_min and self.year_max and self.year_min > self.year_max:
            raise ValueError("year_min 不能大于 year_max")
        return self


class RetrievalQuery(BaseModel):
    """一次检索请求。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1, description="自然语言问题")
    mode: RetrievalMode = "hybrid"
    top_k: int = Field(default=20, ge=1, le=50, description="召回阶段每路取多少")
    max_chunks: int = Field(default=12, ge=1, le=50, description="最终证据块上限")
    max_tokens: int = Field(default=6000, ge=500, le=20000, description="证据包 token 上限")
    expand_hops: int = Field(default=1, ge=0, le=2, description="图扩展跳数")
    filters: QueryFilters = Field(default_factory=QueryFilters)
    as_of: datetime | None = Field(
        default=None, description="时间旅行：只看 recorded_at <= as_of 的记忆"
    )


class Citation(BaseModel):
    """一条可点击的引用。page 是印刷页码，给人看。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    chunk_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_title: str = ""
    page: int = Field(ge=1)
    quote: str = Field(min_length=5, description="原文原句")
    section_path: tuple[str, ...] = ()


class ClaimHit(BaseModel):
    """一条命中断言。必须带 citation，否则不算命中。"""

    model_config = ConfigDict(extra="forbid")

    claim: Claim
    citations: tuple[Citation, ...] = Field(min_length=1)
    channels: tuple[Channel, ...] = Field(default=("lexical",), min_length=1)
    hop: int = Field(default=0, ge=0, le=2, description="0=直接召回，>0=图扩展得到")
    rrf: float = Field(default=0.0, ge=0.0, description="最佳证据块的 RRF 分")
    rank: int = Field(default=0, ge=0)


class ChunkHit(BaseModel):
    """一块原文。text 是截断后的可展示文本，不是整块原文（整块走 read_evidence）。"""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    page: int = Field(ge=1)
    section_path: tuple[str, ...] = ()
    text: str = Field(min_length=1)
    channels: tuple[Channel, ...] = Field(default=("lexical",), min_length=1)
    rrf: float = Field(default=0.0, ge=0.0)
    rank: int = Field(default=0, ge=0)


class NoteHit(BaseModel):
    """命中的笔记。disagrees 的笔记会显著抬高该断言的信息价值。"""

    model_config = ConfigDict(extra="forbid")

    note: Note
    relation: NoteRelation = NoteRelation.CITES
    claim_id: str | None = None


class Conflict(BaseModel):
    """两条断言互相矛盾。只在 contradictions 通道出现，不参与证据排序。"""

    model_config = ConfigDict(extra="forbid")

    claim_a: Claim
    claim_b: Claim
    predicate: Predicate = Predicate.CONTRADICTS
    note: str = ""


class TraceStep(BaseModel):
    """一个阶段的执行记录。整条 trace 就是"可解释"的物证。"""

    model_config = ConfigDict(extra="forbid")

    stage: TraceStage
    elapsed_ms: float = Field(default=0.0, ge=0.0)
    candidates_in: int = Field(default=0, ge=0)
    candidates_out: int = Field(default=0, ge=0)
    detail: dict[str, str | int | float | bool] = Field(default_factory=dict)


class RetrievalStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elapsed_ms: float = Field(default=0.0, ge=0.0)
    tokens_used: int = Field(default=0, ge=0)
    claims_total: int = Field(default=0, ge=0)
    chunks_total: int = Field(default=0, ge=0)
    truncated: bool = False


class RetrievalResult(BaseModel):
    """检索结果。status=empty 是合法结果，必须显式返回，不能返回"看起来相关"的东西。"""

    model_config = ConfigDict(extra="forbid")

    status: ResultStatus
    query: RetrievalQuery
    claims: tuple[ClaimHit, ...] = ()
    chunks: tuple[ChunkHit, ...] = ()
    notes: tuple[NoteHit, ...] = ()
    conflicts: tuple[Conflict, ...] = ()
    trace: tuple[TraceStep, ...] = ()
    stats: RetrievalStats = Field(default_factory=RetrievalStats)
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_status(self) -> RetrievalResult:
        has_any = bool(self.claims or self.chunks or self.notes or self.conflicts)
        if self.status == "ok" and not has_any:
            raise ValueError("status=ok 但结果为空；空结果请显式标为 empty")
        if self.status == "empty" and has_any:
            raise ValueError("status=empty 但存在结果")
        if self.status == "ambiguous" and not has_any:
            raise ValueError("status=ambiguous 但没有任何候选供调用方消歧")
        return self


class Retriever(Protocol):
    """检索器接口。实现方只需要保证：输出满足 RetrievalResult 的约束。"""

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        ...


def estimate_tokens(text: str) -> int:
    """粗估 token：中文按 1 字 1 token，英文按 4 字符 1 token。

    宁可高估：预算超了要截断，低估会撑爆上下文。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return cjk + ((other + 3) // 4)


def fit_budget(
    chunks: Sequence[ChunkHit], max_tokens: int
) -> tuple[list[ChunkHit], bool]:
    """按顺序装填证据块，直到预算用完。返回 (保留下来的块, 是否发生截断)。"""
    kept: list[ChunkHit] = []
    used = 0
    for hit in chunks:
        cost = estimate_tokens(hit.text)
        if kept and used + cost > max_tokens:
            return kept, True
        kept.append(hit)
        used += cost
    return kept, len(kept) < len(chunks)


class RetrievalScore:
    """排序键：不用手调权重，用字典序。

    1) 最佳证据块的 RRF 分（越高越前）
    2) 证据条数（同样相关时，证据多的更可信）
    3) 置信度：EXTRACTED 优先于 INFERRED
    4) claim_id 字典序（保证同一输入的结果稳定可复现）

    明确不用加权和：权重需要标定，标定需要标注集，现在没有。
    """

    __slots__ = ("claim_id", "confidence", "evidence_count", "rrf")

    def __init__(self, rrf: float, evidence_count: int, confidence: Confidence, claim_id: str) -> None:
        self.rrf = rrf
        self.evidence_count = evidence_count
        self.confidence = confidence
        self.claim_id = claim_id

    def sort_key(self) -> tuple[float, int, int, str]:
        return (
            -self.rrf,
            -self.evidence_count,
            0 if self.confidence == Confidence.EXTRACTED else 1,
            self.claim_id,
        )


__all__ = [
    "EXPANSION_PREDICATES",
    "ChunkHit",
    "Citation",
    "ClaimHit",
    "Conflict",
    "NoteHit",
    "QueryFilters",
    "RetrievalQuery",
    "RetrievalResult",
    "RetrievalScore",
    "RetrievalStats",
    "Retriever",
    "TraceStep",
    "estimate_tokens",
    "fit_budget",
]