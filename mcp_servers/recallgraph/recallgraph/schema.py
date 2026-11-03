"""RecallGraph 领域模型（Pydantic v2）。

对应文档：docs/01-数据模型.md。
这里既是"内存里的对象"，也是"入库前的校验层"：所有铁律都在本文件里落成代码。

    from recallgraph.schema import Chunk, Claim, Evidence, ClaimBundle

三条铁律：
1. 版本不可变 —— DocumentVersion / Chunk / Evidence / MemoryEvent 均为 frozen；
2. claim 无证据不生效 —— ClaimBundle 校验 ACTIVE 断言必须有支撑或反驳的证据；
3. 记忆只追加 —— MemoryEvent 只能新建，用 supersedes_id 串联。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from recallgraph.ids import new_ulid, normalize_name
from recallgraph.vocabulary import (
    LITERAL_OBJECT_PREDICATES,
    PREDICATE_SIGNATURES,
    ClaimStatus,
    Confidence,
    DocType,
    EntityKind,
    MemoryOp,
    MemoryStatus,
    NoteRelation,
    Predicate,
    Stance,
    VersionStatus,
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_NUMBER = re.compile(r"^[+-]?\d+(?:[.,]\d+)?%?$")


def utcnow() -> datetime:
    """带时区的当前时间。全库统一用 UTC 存储，展示时再转本地时区。"""
    return datetime.now(UTC)


def normalize_ws(text: str) -> str:
    """折叠所有空白，用于引文与原文的包含性比对。"""
    return " ".join(text.split())


class DomainModel(BaseModel):
    """所有领域对象的基类：禁止多余字段，字符串统一去首尾空白。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ── 第 1 层：身份 ────────────────────────────────────────────────────────────


class Document(DomainModel):
    """论文 / 笔记 / 书 / 网页的"身份"。同一篇论文的多个副本共享一个 Document。"""

    id: str = Field(default_factory=new_ulid)
    title: str = Field(min_length=1)
    title_normalized: str = ""
    authors: list[str] = Field(default_factory=list)
    first_author: str | None = None
    year: int | None = Field(default=None, ge=1900, le=2100)
    external_ids: dict[str, str] = Field(default_factory=dict)
    doc_type: DocType = DocType.PAPER
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _fill_derived(self) -> Self:
        if not self.title_normalized:
            self.title_normalized = normalize_name(self.title)
        if self.first_author is None and self.authors:
            self.first_author = self.authors[0]
        if self.authors:
            self.authors = [a for a in self.authors if a]
        return self


class DocumentVersion(DomainModel):
    """某一版原文。content_hash 唯一：同一份文件重复导入会命中已有版本。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    id: str = Field(default_factory=new_ulid)
    document_id: str = Field(min_length=1)
    content_hash: str = Field(description="全文 SHA-256，64 位小写十六进制")
    file_path: str = Field(min_length=1)
    page_count: int = Field(ge=1)
    published_at: datetime | None = None
    ingested_at: datetime = Field(default_factory=utcnow)
    language: str | None = None
    parser_version: str = Field(min_length=1)
    status: VersionStatus = VersionStatus.ACTIVE

    @model_validator(mode="after")
    def _check_hash(self) -> Self:
        if not _HEX64.match(self.content_hash):
            raise ValueError("content_hash 必须是 64 位小写十六进制（sha256）")
        return self


# ── 第 2 层：原文 ────────────────────────────────────────────────────────────


class Chunk(DomainModel):
    """最小可引用单位。id 由 version_id + seq 确定性推导。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    id: str = ""
    version_id: str = Field(min_length=1)
    seq: int = Field(ge=0)
    page_start: int = Field(ge=1, description="PDF 物理页，从 1 开始")
    page_end: int = Field(ge=1)
    printed_page_start: int | None = Field(default=None, description="论文印刷页码")
    printed_page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)
    text: str = Field(min_length=1)
    context_header: str = Field(min_length=1, description="拼给模型的前缀：标题 + 节名")
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=1)
    token_count: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.page_end < self.page_start:
            raise ValueError("page_end 不能小于 page_start")
        if self.page_end - self.page_start > 1:
            raise ValueError("切块不得跨越超过 1 页，否则引用会失真")
        if self.char_end <= self.char_start:
            raise ValueError("char_end 必须大于 char_start")
        expected = f"{self.version_id}:{self.seq:04d}"
        if self.id and self.id != expected:
            raise ValueError(f"chunk id 必须为 '{expected}'")
        object.__setattr__(self, "id", expected)
        return self


class Entity(DomainModel):
    """概念/方法/数据集/指标/人/机构/论文。对齐只走归一化名 + 别名表。"""

    id: str = Field(default_factory=new_ulid)
    canonical_name: str = Field(min_length=1)
    normalized_name: str = ""
    kind: EntityKind
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _normalize(self) -> Self:
        if not self.normalized_name:
            self.normalized_name = normalize_name(self.canonical_name)
        # 别名按归一化形式去重（保留首次出现的写法）；kind 不同不合并。
        cleaned: list[str] = []
        seen: set[str] = set()
        for alias in [self.canonical_name, *self.aliases]:
            alias = alias.strip()
            key = normalize_name(alias)
            if alias and key and key not in seen:
                seen.add(key)
                cleaned.append(alias)
        self.aliases = cleaned
        return self


class ChunkEntity(DomainModel):
    """段落 ↔ 实体。混合检索命中 chunk 后，靠它映射到图上的实体。"""

    chunk_id: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    mention_text: str = Field(min_length=1)
    char_offset: int = Field(ge=0)


# ── 第 3 层：知识 ────────────────────────────────────────────────────────────


class Claim(DomainModel):
    """一条断言：主语 + 谓词 + 宾语 + 成立条件，一句一条。"""

    id: str = Field(default_factory=new_ulid)
    version_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    predicate: Predicate
    object_id: str | None = None
    object_text: str | None = Field(default=None, description="宾语是字面量时使用")
    conditions: str = Field(default="", description="成立条件；建议尽量填")
    claim_text: str = Field(min_length=5, description="人可读的完整断言，一句")
    confidence: Confidence = Confidence.EXTRACTED
    status: ClaimStatus = ClaimStatus.DRAFT
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _check_object(self) -> Self:
        is_literal = self.predicate in LITERAL_OBJECT_PREDICATES
        if is_literal:
            if self.object_id is not None:
                raise ValueError(f"{self.predicate} 的宾语必须是字面量，不能挂实体")
            if not self.object_text or not self.object_text.strip():
                raise ValueError(f"{self.predicate} 必须提供 object_text（数值等字面量）")
            if not _NUMBER.match(self.object_text.strip()):
                raise ValueError(f"{self.predicate} 的 object_text 必须是数值，收到 {self.object_text!r}")
        else:
            if self.object_id is None:
                raise ValueError(f"{self.predicate} 必须挂实体宾语（object_id）")
            if self.object_id == self.subject_id:
                raise ValueError("主语与宾语不能是同一实体")
        if "\n" in self.claim_text:
            raise ValueError("claim_text 必须是一句话，不能含换行")
        return self


class Evidence(DomainModel):
    """证据：把断言钉回原句。quote 必须能在 chunk 原文里逐字找到。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    id: str = Field(default_factory=new_ulid)
    claim_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=5, description="原文原句，逐字复制")
    page: int = Field(ge=1, description="印刷页码，给人看")
    stance: Stance = Stance.SUPPORT
    created_at: datetime = Field(default_factory=utcnow)


# ── 第 4 层：笔记与记忆 ─────────────────────────────────────────────────────


class Note(DomainModel):
    """人对知识的加工。author=user 表示"我说的"，用于和论文结论区分。"""

    id: str = Field(default_factory=new_ulid)
    author: str = Field(default="user", min_length=1)
    title: str = Field(min_length=1)
    body: str = Field(min_length=1)
    linked_version_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime | None = None


class NoteClaim(DomainModel):
    """笔记 ↔ 断言。disagrees 是本项目的核心动作。"""

    note_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    relation: NoteRelation = NoteRelation.CITES


class MemoryEvent(DomainModel):
    """记忆事件：只追加。改记忆 = 新事件 + 旧事件标 superseded。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    id: str = Field(default_factory=new_ulid)
    user_id: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    memory_key: str = Field(min_length=1)
    operation: MemoryOp
    value: str = Field(min_length=1)
    valid_from: datetime
    valid_to: datetime | None = None
    recorded_at: datetime = Field(default_factory=utcnow)
    source_turn_id: str | None = None
    source_chunk_id: str | None = None
    supersedes_id: str | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.operation in (MemoryOp.UPDATE, MemoryOp.RETRACT) and not self.supersedes_id:
            raise ValueError(f"{self.operation} 必须提供 supersedes_id")
        if self.operation == MemoryOp.ADD and self.supersedes_id:
            raise ValueError("ADD 不应带 supersedes_id；改写请用 UPDATE")
        if self.valid_to is not None and self.valid_to < self.valid_from:
            raise ValueError("valid_to 不能早于 valid_from")
        if self.operation == MemoryOp.RETRACT and self.status != MemoryStatus.RETRACTED:
            raise ValueError("RETRACT 事件的 status 必须是 retracted")
        if self.valid_to is not None and self.status == MemoryStatus.ACTIVE:
            raise ValueError("valid_to 已设置的事件不可能是 active，请标为 superseded/retracted")
        return self


# ── 聚合校验：把跨对象铁律落成代码 ──────────────────────────────────────────


class ClaimBundle(DomainModel):
    """一次导入产出的完整知识包，用于入库前做一致性校验。

    校验项：
    - 引文必须能在对应 chunk 原文里找到（防抽取编造）；
    - ACTIVE 断言必须至少有一条 support/refute 证据；
    - 谓词签名必须匹配（主语/宾语的 kind 在词表允许范围内）；
    - 交叉引用的 id 必须存在。
    """

    chunks: Sequence[Chunk] = Field(default_factory=tuple)
    entities: Sequence[Entity] = Field(default_factory=tuple)
    claims: Sequence[Claim] = Field(default_factory=tuple)
    evidence: Sequence[Evidence] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _check_bundle(self) -> Self:
        chunks = {c.id: c for c in self.chunks}
        entities = {e.id: e for e in self.entities}
        claims = {c.id: c for c in self.claims}
        errors: list[str] = []

        for ev in self.evidence:
            chunk = chunks.get(ev.chunk_id)
            if chunk is None:
                errors.append(f"evidence {ev.id} 指向不存在的 chunk {ev.chunk_id}")
                continue
            if ev.claim_id not in claims:
                errors.append(f"evidence {ev.id} 指向不存在的 claim {ev.claim_id}")
            if normalize_ws(ev.quote) not in normalize_ws(chunk.text):
                errors.append(f"evidence {ev.id} 的引文在 chunk {ev.chunk_id} 中找不到：{ev.quote[:40]!r}")

        by_claim: dict[str, list[Evidence]] = {}
        for ev in self.evidence:
            by_claim.setdefault(ev.claim_id, []).append(ev)

        for claim in self.claims:
            subject = entities.get(claim.subject_id)
            if subject is None:
                errors.append(f"claim {claim.id} 的主语实体 {claim.subject_id} 不存在")
                continue
            object_kind: EntityKind | None = None
            if claim.object_id is not None:
                obj = entities.get(claim.object_id)
                if obj is None:
                    errors.append(f"claim {claim.id} 的宾语实体 {claim.object_id} 不存在")
                    continue
                object_kind = obj.kind

            allowed_subjects, allowed_objects = PREDICATE_SIGNATURES[claim.predicate]
            if subject.kind not in allowed_subjects:
                errors.append(
                    f"claim {claim.id} 谓词 {claim.predicate} 不接受主语类型 {subject.kind}"
                )
            if allowed_objects is None:
                if object_kind is not None:
                    errors.append(f"claim {claim.id} 谓词 {claim.predicate} 只接受字面量宾语")
            elif object_kind is not None and object_kind not in allowed_objects:
                errors.append(
                    f"claim {claim.id} 谓词 {claim.predicate} 不接受宾语类型 {object_kind}"
                )

            if claim.status == ClaimStatus.ACTIVE:
                evs = by_claim.get(claim.id, [])
                if not any(e.stance in (Stance.SUPPORT, Stance.REFUTE) for e in evs):
                    errors.append(f"claim {claim.id} 状态为 active 但没有任何支撑/反驳证据")

        if errors:
            raise ValueError("数据包校验失败：\n- " + "\n- ".join(errors))
        return self

    def active_claims(self) -> Iterable[Claim]:
        return (c for c in self.claims if c.status == ClaimStatus.ACTIVE)

    def evidence_for(self, claim_id: str) -> list[Evidence]:
        return [e for e in self.evidence if e.claim_id == claim_id]


__all__ = [
    "Chunk",
    "ChunkEntity",
    "Claim",
    "ClaimBundle",
    "Document",
    "DocumentVersion",
    "Entity",
    "Evidence",
    "MemoryEvent",
    "Note",
    "NoteClaim",
    "normalize_ws",
    "utcnow",
]