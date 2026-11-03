"""抽取层测试：契约硬约束 + 桩规则 + 落库幂等。

最重要的一条是"引文必须是原文子串"——这条如果松了，后面的页码引用、
证据核对、评测里的 citation accuracy 就全是假的。
"""

from __future__ import annotations

import psycopg
import pytest

from recallgraph.extract import get_extractor
from recallgraph.extract.contract import (
    ClaimDraft,
    EvidenceDraft,
    ExtractionError,
    ExtractionResult,
    check_claim,
    validate_result,
)
from recallgraph.extract.stub import StubExtractor
from recallgraph.extract.writer import extract_version
from recallgraph.ingest.pipeline import ingest
from recallgraph.schema import Chunk
from recallgraph.storage import connect, ensure_schema, load_ingest, schema_ready
from recallgraph.vocabulary import EntityKind, Predicate


def chunk(text: str, *, page: int = 1, page_end: int | None = None, seq: int = 1) -> Chunk:
    return Chunk(
        id=f"ver-1:{seq:04d}",
        version_id="ver-1",
        seq=seq,
        page_start=page,
        page_end=page_end if page_end is not None else page,
        section_path=["Method"],
        text=text,
        context_header="Test Paper ｜ Method",
        char_start=0,
        char_end=len(text),
        token_count=max(1, len(text) // 2),
    )


# ── 契约 ───────────────────────────────────────────────────────────────────


def test_quote_must_be_verbatim_substring() -> None:
    good = ClaimDraft(
        subject="RAG",
        predicate=Predicate.IMPROVES,
        object="factual accuracy",
        claim_text="RAG improves factual accuracy",
        evidence=[EvidenceDraft(chunk_id="ver-1:0001", quote="RAG improves factual accuracy", page=1)],
    )
    check_claim(good, {"ver-1:0001": chunk("RAG improves factual accuracy on open-domain QA.")})

    bad = good.model_copy(
        update={"evidence": [EvidenceDraft(chunk_id="ver-1:0001", quote="RAG improves precision", page=1)]}
    )
    with pytest.raises(ExtractionError, match="不是原文子串"):
        check_claim(bad, {"ver-1:0001": chunk("RAG improves factual accuracy on open-domain QA.")})


def test_quote_matching_tolerates_whitespace_and_width() -> None:
    text = "RAG improves\nfactual accuracy."
    claim = ClaimDraft(
        subject="RAG",
        predicate=Predicate.IMPROVES,
        object="factual accuracy",
        claim_text="RAG improves factual accuracy",
        evidence=[EvidenceDraft(chunk_id="ver-1:0001", quote="RAG improves factual accuracy", page=1)],
    )
    check_claim(claim, {"ver-1:0001": chunk(text)})


def test_page_outside_chunk_span_is_rejected() -> None:
    claim = ClaimDraft(
        subject="RAG",
        predicate=Predicate.IMPROVES,
        object="accuracy",
        claim_text="RAG improves accuracy",
        evidence=[EvidenceDraft(chunk_id="ver-1:0001", quote="RAG improves accuracy", page=9)],
    )
    with pytest.raises(ExtractionError, match="不在块"):
        check_claim(claim, {"ver-1:0001": chunk("RAG improves accuracy.", page=1, page_end=2)})


def test_bad_claim_does_not_kill_the_batch() -> None:
    result = ExtractionResult(
        claims=[
            ClaimDraft(
                subject="A",
                predicate=Predicate.USES,
                object="B",
                claim_text="A uses B",
                evidence=[EvidenceDraft(chunk_id="ver-1:0001", quote="A uses B", page=1)],
            ),
            ClaimDraft(
                subject="C",
                predicate=Predicate.USES,
                object="D",
                claim_text="C uses D",
                evidence=[EvidenceDraft(chunk_id="ver-1:0001", quote="C uses X", page=1)],
            ),
        ]
    )
    passed, errors = validate_result(result, [chunk("A uses B for retrieval.")])
    assert len(passed) == 1
    assert len(errors) == 1


def test_achieves_needs_literal_object() -> None:
    with pytest.raises(ValueError):
        ClaimDraft(
            subject="ResNet-50",
            predicate=Predicate.ACHIEVES,
            object=None,
            claim_text="ResNet-50 achieves",
            evidence=[EvidenceDraft(chunk_id="ver-1:0001", quote="ResNet-50", page=1)],
        )


# ── 桩抽取器 ───────────────────────────────────────────────────────────────


def test_stub_extracts_simple_assertion() -> None:
    extractor = StubExtractor()
    result = extractor.extract(chunk("Dense retrieval improves recall over sparse retrieval in open-domain QA."))
    assert result.claims, "至少应抽出一条"
    claim = result.claims[0]
    assert claim.predicate is Predicate.IMPROVES
    assert "recall" in (claim.object or "")
    assert claim.confidence.value == "EXTRACTED"


def test_stub_ignores_pronoun_subjects_and_numbers() -> None:
    extractor = StubExtractor()
    result = extractor.extract(chunk("It improves accuracy by 3 points. 42 17 23 91 8 6 4."))
    assert result.claims == []


def test_stub_handles_chinese_sentences() -> None:
    extractor = StubExtractor()
    result = extractor.extract(chunk("稠密检索提升了开放域问答的召回率，但依赖标注数据。"))
    assert result.claims
    assert result.claims[0].predicate is Predicate.IMPROVES


def test_stub_page_estimate_stays_in_span() -> None:
    extractor = StubExtractor()
    body = "Filler sentence without any verb pattern. " * 20 + "RAG improves factual accuracy."
    result = extractor.extract(chunk(body, page=3, page_end=4))
    claim = result.claims[0]
    assert 3 <= claim.evidence[0].page <= 4


def test_unknown_extractor_raises() -> None:
    with pytest.raises(KeyError):
        get_extractor("nope")


def test_entity_kind_heuristic() -> None:
    extractor = StubExtractor()
    result = extractor.extract(chunk("BEIR is a benchmark. RAG improves accuracy on Natural Questions."))
    kinds = {entity.canonical_name: entity.kind for entity in result.entities}
    assert kinds["RAG"] is EntityKind.METHOD
    assert any(kind is EntityKind.DATASET for kind in kinds.values())


# ── 落库 ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def db():
    try:
        conn = connect()
    except psycopg.OperationalError:
        pytest.skip("PostgreSQL 不可用：先执行 docker compose up -d")
    ensure_schema(conn)
    if not schema_ready(conn):
        pytest.skip("表结构未就绪")
    yield conn
    conn.close()


@pytest.fixture()
def tx(db):
    with db.transaction(force_rollback=True):
        yield db


def _version(tx, tmp_path, body: str) -> str:
    path = tmp_path / "p.md"
    path.write_text(body, encoding="utf-8")
    return load_ingest(tx, ingest(path)).version_id


def test_extract_writes_claims_evidence_and_links(tx, tmp_path) -> None:
    version_id = _version(
        tx,
        tmp_path,
        "# RAG Paper\n\n## Method\n\n"
        "Dense retrieval improves recall over sparse retrieval. "
        "The retriever uses a dual encoder. The reader generates answers.\n",
    )
    stats = extract_version(tx, version_id, StubExtractor())
    assert stats.claims_written >= 2
    assert stats.evidence_written >= stats.claims_written
    assert stats.errors == []

    count = tx.execute(
        "SELECT count(*) FROM claims WHERE version_id = %s AND status = 'active'", (version_id,)
    ).fetchone()[0]
    assert count == stats.claims_written

    links = tx.execute("SELECT count(*) FROM chunk_entities ce JOIN chunks c ON c.id = ce.chunk_id "
                       "WHERE c.version_id = %s", (version_id,)).fetchone()[0]
    assert links >= 1


def test_extract_is_idempotent_unless_forced(tx, tmp_path) -> None:
    version_id = _version(tx, tmp_path, "# Paper\n\n## Method\n\nRAG improves accuracy.\n")
    first = extract_version(tx, version_id, StubExtractor())
    assert first.claims_written >= 1

    second = extract_version(tx, version_id, StubExtractor())
    assert second.claims_written == 0
    assert "已抽取过" in second.skipped_reason


def test_force_retracts_old_claims_but_keeps_rows(tx, tmp_path) -> None:
    version_id = _version(tx, tmp_path, "# Paper\n\n## Method\n\nRAG improves accuracy.\n")
    extract_version(tx, version_id, StubExtractor())
    stats = extract_version(tx, version_id, StubExtractor(), force=True)
    assert stats.retracted >= 1

    rows = tx.execute(
        "SELECT status, count(*) FROM claims WHERE version_id = %s GROUP BY status", (version_id,)
    ).fetchall()
    by_status = dict(rows)
    assert by_status.get("retracted", 0) >= 1
    assert by_status.get("active", 0) == stats.claims_written


def test_extract_missing_version_is_reported(tx) -> None:
    stats = extract_version(tx, "01ZZZZZZZZZZZZZZZZZZZZZZZZZ", StubExtractor())
    assert stats.claims_written == 0
    assert "不存在" in stats.skipped_reason

def test_achieves_requires_numeric_object():
    """模型会把"197 项测试通过"这种句子当 achieves 的宾语，契约要挡住。"""
    with pytest.raises(ValueError, match="必须是数值"):
        ClaimDraft(
            subject="测试",
            predicate=Predicate.ACHIEVES,
            object="197 项测试通过",
            claim_text="项目达到 197 项测试通过",
            evidence=[EvidenceDraft(chunk_id="c", quote="197 项测试通过", page=1)],
        )


def test_single_failing_chunk_does_not_abort_version(tx, tmp_path) -> None:
    """一段抽不出来（模型超时/JSON 截断）时，其余段落照写，失败记进 stats.errors。"""
    version_id = _version(
        tx,
        tmp_path,
        "# Paper\n\n## Method\n\n"
        + "Dense retrieval improves recall over sparse retrieval. " * 26
        + "\n\nThe retriever uses a dual encoder. " * 12
        + "\n",
    )

    class Flaky(StubExtractor):
        calls = 0

        def extract(self, chunk):
            Flaky.calls += 1
            if "dual encoder" in chunk.text:
                raise ExtractionError("模拟接口截断")
            return super().extract(chunk)

    stats = extract_version(tx, version_id, Flaky())
    assert Flaky.calls >= 2  # 失败的那段没有打断后续段落
    assert stats.claims_written == 1  # 成功的那段照常写入
    assert stats.errors and all("模拟接口截断" in e for e in stats.errors)
