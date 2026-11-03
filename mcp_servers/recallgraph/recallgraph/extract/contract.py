"""抽取契约：模型（或桩）交出来的东西，必须满足这些形状才能入库。

为什么先定契约再接模型：模型会变（今天桩、明天 DeepSeek、后天本地模型），
但"一条断言长什么样"不该跟着变。契约里最硬的一条是**证据必须能在原文里找到**——
引文不是原文子串，直接拒绝，不给下游"看着像"的机会。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from pydantic import BaseModel, Field, model_validator

from recallgraph.schema import Chunk
from recallgraph.vocabulary import Confidence, EntityKind, Predicate, Stance

MIN_QUOTE_CHARS = 5
# 与 schema.Claim 的数值规则保持一致：achieves 的宾语是字面量，必须像个数
_NUMBER = re.compile(r"^[+-]?\d+(?:[.,]\d+)?%?$")


class ExtractionError(ValueError):
    """抽取结果不符合契约。消息要能直接告诉人哪里错了。"""


class EvidenceDraft(BaseModel):
    chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=MIN_QUOTE_CHARS, description="原文原句，逐字复制")
    page: int = Field(ge=1)
    stance: Stance = Stance.SUPPORT


class ClaimDraft(BaseModel):
    subject: str = Field(min_length=1, max_length=120)
    predicate: Predicate
    object: str | None = Field(default=None, max_length=120)
    conditions: str = ""
    claim_text: str = Field(min_length=5, max_length=400)
    confidence: Confidence = Confidence.EXTRACTED
    evidence: list[EvidenceDraft] = Field(min_length=1)

    @model_validator(mode="after")
    def _object_rules(self) -> ClaimDraft:
        if self.predicate is Predicate.ACHIEVES:
            if not self.object:
                raise ValueError("achieves 的宾语必须是字面量（如 76.8% top-1），不能为空")
            if not _NUMBER.match(self.object.strip()):
                raise ValueError(f"achieves 的宾语必须是数值，收到 {self.object!r}")
        elif not self.object:
            raise ValueError(f"{self.predicate.value} 必须给出宾语实体")
        if self.object and self.object.strip() == self.subject.strip():
            raise ValueError("主语与宾语相同")
        return self


class EntityDraft(BaseModel):
    canonical_name: str = Field(min_length=1, max_length=120)
    kind: EntityKind = EntityKind.CONCEPT
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None


class ExtractionResult(BaseModel):
    entities: list[EntityDraft] = Field(default_factory=list)
    claims: list[ClaimDraft] = Field(default_factory=list)


@dataclass(frozen=True)
class CheckedClaim:
    """通过校验、可以直接写库的断言。"""

    draft: ClaimDraft
    evidence_pages: tuple[int, ...]


_WS = re.compile(r"\s+")


def normalize_for_match(text: str) -> str:
    """引文比对用：全半角统一、空白折叠、大小写不敏感。"""
    return _WS.sub(" ", unicodedata.normalize("NFKC", text)).strip().casefold()


def check_claim(claim: ClaimDraft, chunks: dict[str, Chunk]) -> CheckedClaim:
    """逐条证据核对：块存在、引文是原文子串、页码落在块的页码范围里。"""
    for evidence in claim.evidence:
        chunk = chunks.get(evidence.chunk_id)
        if chunk is None:
            raise ExtractionError(f"证据指向不存在的块：{evidence.chunk_id}")

        if normalize_for_match(evidence.quote) not in normalize_for_match(chunk.text):
            raise ExtractionError(
                f"引文不是原文子串（块 {chunk.id}）：{evidence.quote[:40]!r}"
            )

        if not (chunk.page_start <= evidence.page <= chunk.page_end):
            raise ExtractionError(
                f"页码 {evidence.page} 不在块 {chunk.id} 的范围 p{chunk.page_start}-p{chunk.page_end}"
            )

    return CheckedClaim(draft=claim, evidence_pages=tuple(e.page for e in claim.evidence))


def validate_result(result: ExtractionResult, chunks: list[Chunk]) -> tuple[list[CheckedClaim], list[str]]:
    """批量校验，返回（通过的断言, 报错信息）。坏一条不拖垮整批，但必须报告。"""
    by_id = {chunk.id: chunk for chunk in chunks}
    passed: list[CheckedClaim] = []
    errors: list[str] = []
    for claim in result.claims:
        try:
            passed.append(check_claim(claim, by_id))
        except ExtractionError as exc:
            errors.append(f"{claim.claim_text[:40]!r}: {exc}")
    return passed, errors