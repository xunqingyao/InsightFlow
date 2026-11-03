"""受控词表：实体类别、谓词、状态枚举，以及谓词的类型签名。

设计原则：
- 枚举一旦发布只增不改，改名靠迁移而不是重命名；
- 每个谓词都有 (主语 kind, 宾语 kind) 签名，签不上名的断言不进库；
- 词表是"最小可覆盖评测集"的集合，加词必须有题目撑腰。
"""

from __future__ import annotations

from enum import StrEnum


class DocType(StrEnum):
    """文档类型。"""

    PAPER = "paper"
    NOTE = "note"
    BOOK = "book"
    WEB = "web"


class VersionStatus(StrEnum):
    """版本状态。版本不可变，"过期"用状态表达。"""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    FAILED = "failed"


class EntityKind(StrEnum):
    """实体类别。论文本身也是实体，用于承载 proposes / evaluates_on 的主语。"""

    METHOD = "method"
    DATASET = "dataset"
    METRIC = "metric"
    CONCEPT = "concept"
    PERSON = "person"
    ORG = "org"
    PAPER = "paper"


class Predicate(StrEnum):
    """谓词受控词表 v1：12 个。"""

    IMPROVES = "improves"
    OUTPERFORMS = "outperforms"
    ACHIEVES = "achieves"
    REPLACES = "replaces"
    REQUIRES = "requires"
    USES = "uses"
    CAUSES = "causes"
    IS_A = "is_a"
    PART_OF = "part_of"
    PROPOSES = "proposes"
    EVALUATES_ON = "evaluates_on"
    CONTRADICTS = "contradicts"


class Confidence(StrEnum):
    """断言来源等级。EXTRACTED=原文显式，INFERRED=模型推断。"""

    EXTRACTED = "EXTRACTED"
    INFERRED = "INFERRED"


class ClaimStatus(StrEnum):
    """断言状态。draft 不进检索。"""

    DRAFT = "draft"
    ACTIVE = "active"
    RETRACTED = "retracted"


class Stance(StrEnum):
    """证据对断言的态度。"""

    SUPPORT = "support"
    REFUTE = "refute"
    CONTEXT = "context"


class NoteRelation(StrEnum):
    """笔记与断言的关系。disagrees 是本项目的核心动作。"""

    AGREES = "agrees"
    DISAGREES = "disagrees"
    EXTENDS = "extends"
    CITES = "cites"


class MemoryOp(StrEnum):
    """记忆操作。只追加，不改写。"""

    ADD = "add"
    UPDATE = "update"
    RETRACT = "retract"
    NOOP = "noop"


class MemoryStatus(StrEnum):
    """记忆事件状态。"""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"


# 谓词签名：(允许的主语 kind 集合, 允许的宾语 kind 集合)。
# 宾语集合为 None 表示"宾语必须是字面量"（写入 object_text，不挂实体）。
PREDICATE_SIGNATURES: dict[Predicate, tuple[frozenset[EntityKind], frozenset[EntityKind] | None]] = {
    Predicate.IMPROVES: (
        frozenset({EntityKind.METHOD}),
        frozenset({EntityKind.METRIC, EntityKind.CONCEPT}),
    ),
    Predicate.OUTPERFORMS: (
        frozenset({EntityKind.METHOD}),
        frozenset({EntityKind.METHOD}),
    ),
    Predicate.ACHIEVES: (
        frozenset({EntityKind.METHOD}),
        None,  # 数值：78.4 / 61 / "+12.3"
    ),
    Predicate.REPLACES: (
        frozenset({EntityKind.METHOD}),
        frozenset({EntityKind.METHOD}),
    ),
    Predicate.REQUIRES: (
        frozenset({EntityKind.METHOD}),
        frozenset({EntityKind.CONCEPT, EntityKind.DATASET, EntityKind.METHOD}),
    ),
    Predicate.USES: (
        frozenset({EntityKind.METHOD}),
        frozenset({EntityKind.METHOD, EntityKind.DATASET, EntityKind.CONCEPT}),
    ),
    Predicate.CAUSES: (
        frozenset({EntityKind.METHOD, EntityKind.CONCEPT}),
        frozenset({EntityKind.CONCEPT}),
    ),
    Predicate.IS_A: (
        frozenset({EntityKind.METHOD, EntityKind.DATASET, EntityKind.METRIC, EntityKind.CONCEPT}),
        frozenset({EntityKind.CONCEPT}),
    ),
    Predicate.PART_OF: (
        frozenset({EntityKind.METHOD, EntityKind.DATASET, EntityKind.CONCEPT}),
        frozenset({EntityKind.METHOD, EntityKind.CONCEPT}),
    ),
    Predicate.PROPOSES: (
        frozenset({EntityKind.PERSON, EntityKind.ORG, EntityKind.PAPER}),
        frozenset({EntityKind.METHOD}),
    ),
    Predicate.EVALUATES_ON: (
        frozenset({EntityKind.PAPER, EntityKind.METHOD}),
        frozenset({EntityKind.DATASET, EntityKind.CONCEPT}),
    ),
    Predicate.CONTRADICTS: (
        frozenset({EntityKind.METHOD, EntityKind.CONCEPT, EntityKind.PAPER}),
        frozenset({EntityKind.METHOD, EntityKind.CONCEPT, EntityKind.PAPER}),
    ),
}

# 宾语为字面量的谓词。其余谓词必须挂实体。
LITERAL_OBJECT_PREDICATES: frozenset[Predicate] = frozenset({Predicate.ACHIEVES})