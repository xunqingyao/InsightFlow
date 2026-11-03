"""schema.py 的契约测试：三条铁律 + 谓词签名 + 引文可核验。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from recallgraph.ids import new_ulid, normalize_name
from recallgraph.schema import (
    Chunk,
    Claim,
    ClaimBundle,
    Document,
    DocumentVersion,
    Entity,
    Evidence,
    MemoryEvent,
    normalize_ws,
)
from recallgraph.vocabulary import (
    ClaimStatus,
    Confidence,
    EntityKind,
    MemoryOp,
    MemoryStatus,
    Predicate,
    Stance,
)

NOW = datetime(2026, 10, 27, tzinfo=UTC)
HASH = "a" * 64


def make_version() -> DocumentVersion:
    return DocumentVersion(
        document_id=new_ulid(),
        content_hash=HASH,
        file_path="/corpus/rag.pdf",
        page_count=12,
        parser_version="pypdf-0.1",
    )


def make_chunk(version: DocumentVersion, seq: int = 0, text: str | None = None) -> Chunk:
    return Chunk(
        version_id=version.id,
        seq=seq,
        page_start=5,
        page_end=5,
        printed_page_start=4,
        printed_page_end=4,
        section_path=["3", "3.2", "Retrieval"],
        text=text or "RAG outperforms fine-tuning on open-domain QA by a large margin.",
        context_header="3 Method > 3.2 Retrieval",
        char_start=1200,
        char_end=1268,
    )


def make_entities() -> dict[str, Entity]:
    return {
        "rag": Entity(canonical_name="Retrieval-Augmented Generation", kind=EntityKind.METHOD,
                      aliases=["RAG"]),
        "ft": Entity(canonical_name="fine-tuning", kind=EntityKind.METHOD),
        "qa": Entity(canonical_name="open-domain QA", kind=EntityKind.CONCEPT),
    }


def make_claim(subject: Entity, predicate: Predicate, obj: Entity | None,
               *, status: ClaimStatus = ClaimStatus.ACTIVE, object_text: str | None = None) -> Claim:
    return Claim(
        version_id=new_ulid(),
        subject_id=subject.id,
        predicate=predicate,
        object_id=obj.id if obj else None,
        object_text=object_text,
        claim_text="RAG 在开放域问答上优于参数微调",
        confidence=Confidence.EXTRACTED,
        status=status,
    )


# ── 段落 ───────────────────────────────────────────────────────────────────


def test_chunk_id_is_deterministic() -> None:
    version = make_version()
    chunk = make_chunk(version, seq=7)
    assert chunk.id == f"{version.id}:0007"


def test_chunk_rejects_multipage_span() -> None:
    version = make_version()
    with pytest.raises(ValidationError, match="切块不得跨越超过 1 页"):
        Chunk(
            version_id=version.id,
            seq=0,
            page_start=3,
            page_end=5,
            text="x" * 10,
            context_header="h",
            char_start=0,
            char_end=10,
        )


def test_chunk_is_frozen() -> None:
    chunk = make_chunk(make_version())
    with pytest.raises(ValidationError):
        chunk.text = "tampered"  # type: ignore[misc]


# ── 身份 ───────────────────────────────────────────────────────────────────


def test_document_derives_normalized_title_and_author() -> None:
    doc = Document(title="  Retrieval-Augmented Generation  ", authors=["Lewis", "Perez"])
    assert doc.title_normalized == "retrieval-augmented generation"
    assert doc.first_author == "Lewis"


def test_entity_normalization_and_alias_dedupe() -> None:
    ent = Entity(canonical_name="ＲＡＧ", kind=EntityKind.METHOD, aliases=["rag", "RAG", "检索增强"])
    assert ent.normalized_name == "rag"
    # 归一化后与标准名相同的写法被折叠，只保留标准名 + 真正不同的别名
    assert ent.aliases == ["ＲＡＧ", "检索增强"]


def test_normalize_name_is_deterministic() -> None:
    assert normalize_name("  RAG，") == "rag"
    assert normalize_name("Retrieval   Augmented") == "retrieval augmented"


# ── 断言 ───────────────────────────────────────────────────────────────────


def test_claim_achieves_needs_numeric_literal() -> None:
    ents = make_entities()
    claim = make_claim(ents["rag"], Predicate.ACHIEVES, None, object_text="78.4%")
    assert claim.object_id is None
    with pytest.raises(ValidationError, match="必须是数值"):
        make_claim(ents["rag"], Predicate.ACHIEVES, None, object_text="很好")


def test_claim_achieves_rejects_entity_object() -> None:
    ents = make_entities()
    with pytest.raises(ValidationError, match="必须是字面量"):
        make_claim(ents["rag"], Predicate.ACHIEVES, ents["qa"])


def test_claim_requires_object_for_relational_predicate() -> None:
    ents = make_entities()
    with pytest.raises(ValidationError, match="必须挂实体宾语"):
        make_claim(ents["rag"], Predicate.OUTPERFORMS, None)


def test_claim_rejects_self_loop() -> None:
    ents = make_entities()
    with pytest.raises(ValidationError, match="同一实体"):
        make_claim(ents["rag"], Predicate.OUTPERFORMS, ents["rag"])


# ── 知识包 ─────────────────────────────────────────────────────────────────


def build_bundle(*, quote: str | None = None, status: ClaimStatus = ClaimStatus.ACTIVE,
                 mutate=None) -> ClaimBundle:
    version = make_version()
    chunk = make_chunk(version)
    ents = make_entities()
    claim = make_claim(ents["rag"], Predicate.OUTPERFORMS, ents["ft"], status=status)
    evidence = Evidence(
        claim_id=claim.id,
        chunk_id=chunk.id,
        quote=quote or "RAG outperforms fine-tuning",
        page=4,
        stance=Stance.SUPPORT,
    )
    bundle = ClaimBundle(
        chunks=[chunk],
        entities=list(ents.values()),
        claims=[claim],
        evidence=[evidence],
    )
    if mutate:
        mutate(bundle)
    return bundle


def test_bundle_accepts_valid_package() -> None:
    bundle = build_bundle()
    assert [c.id for c in bundle.active_claims()] == [bundle.claims[0].id]
    assert len(bundle.evidence_for(bundle.claims[0].id)) == 1


def test_bundle_rejects_fabricated_quote() -> None:
    with pytest.raises(ValidationError, match="引文在 chunk"):
        build_bundle(quote="RAG beats fine-tuning by 40 points")


def test_bundle_rejects_active_claim_without_evidence() -> None:
    version = make_version()
    chunk = make_chunk(version)
    ents = make_entities()
    claim = make_claim(ents["rag"], Predicate.OUTPERFORMS, ents["ft"])
    with pytest.raises(ValidationError, match="没有任何支撑/反驳证据"):
        ClaimBundle(chunks=[chunk], entities=list(ents.values()), claims=[claim], evidence=[])


def test_bundle_rejects_wrong_predicate_signature() -> None:
    version = make_version()
    chunk = make_chunk(version)
    ents = make_entities()
    # outperforms 只接受 method 作主语；这里故意用 concept
    claim = make_claim(ents["qa"], Predicate.OUTPERFORMS, ents["ft"])
    evidence = Evidence(claim_id=claim.id, chunk_id=chunk.id,
                        quote="RAG outperforms fine-tuning", page=4)
    with pytest.raises(ValidationError, match="不接受主语类型"):
        ClaimBundle(chunks=[chunk], entities=list(ents.values()),
                    claims=[claim], evidence=[evidence])


def test_quote_comparison_ignores_whitespace() -> None:
    assert normalize_ws("a\n b   c") == "a b c"


# ── 记忆 ───────────────────────────────────────────────────────────────────


def make_memory(**overrides) -> MemoryEvent:
    base = {
        "user_id": "u1",
        "namespace": "research/rag",
        "memory_key": "rag_vs_finetune",
        "operation": MemoryOp.ADD,
        "value": "RAG 在开放域问答上更优",
        "valid_from": NOW,
        "recorded_at": NOW,
    }
    base.update(overrides)
    return MemoryEvent(**base)


def test_memory_add_ok() -> None:
    ev = make_memory()
    assert ev.status == MemoryStatus.ACTIVE


def test_memory_update_requires_supersedes() -> None:
    with pytest.raises(ValidationError, match="supersedes_id"):
        make_memory(operation=MemoryOp.UPDATE)


def test_memory_add_rejects_supersedes() -> None:
    with pytest.raises(ValidationError, match="ADD 不应带"):
        make_memory(supersedes_id=new_ulid())


def test_memory_retract_requires_retracted_status() -> None:
    with pytest.raises(ValidationError, match="status 必须是 retracted"):
        make_memory(operation=MemoryOp.RETRACT, supersedes_id=new_ulid())


def test_memory_rejects_inverted_validity() -> None:
    with pytest.raises(ValidationError, match="valid_to 不能早于"):
        make_memory(valid_to=NOW - timedelta(days=1))


# ── ID ─────────────────────────────────────────────────────────────────────


def test_ulid_format_and_uniqueness() -> None:
    ids = {new_ulid() for _ in range(200)}
    assert len(ids) == 200
    assert all(len(i) == 26 for i in ids)
    assert all(set(i) <= set("0123456789ABCDEFGHJKMNPQRSTVWXYZ") for i in ids)