"""检索层测试：断言证据包必须是"可核对"的，不是"看起来相关"。

这里只测已经实现的两条通道（关键词 + 图扩展）。向量通道还没接，
所以测试里不假设它存在——缺的通道靠 warnings 说出来。
"""

from __future__ import annotations

import psycopg
import pytest

from recallgraph.extract import get_extractor
from recallgraph.extract.contract import normalize_for_match
from recallgraph.extract.writer import extract_version
from recallgraph.ingest.pipeline import ingest
from recallgraph.retrieval import QueryFilters, RetrievalQuery
from recallgraph.retriever import RERANK_HEAD, SqlRetriever
from recallgraph.storage import connect, ensure_schema, load_ingest, schema_ready

BODY = """# Retrieval Notes

## Training

RAG 的训练是端到端微调，文档编码器和查询编码器保持冻结。
The retriever has no independent supervision signal in end-to-end training.

## Results

RAG improves factual accuracy on open-domain question answering benchmarks.
"""


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
def seeded(db, tmp_path):
    """一个独立事务：灌一篇笔记 → 抽断言 → 检索，结束回滚。"""
    path = tmp_path / "notes.md"
    path.write_text(BODY, encoding="utf-8")
    with db.transaction(force_rollback=True):
        version_id = load_ingest(db, ingest(path)).version_id
        extract_version(db, version_id, get_extractor("stub"))
        yield db, version_id


def test_retrieval_returns_claims_with_citations(seeded) -> None:
    conn, _ = seeded
    result = SqlRetriever(conn, rerank=False).retrieve(RetrievalQuery(text="retriever supervision training", max_chunks=5))

    assert result.status == "ok"
    assert result.claims, "应命中至少一条断言"
    for hit in result.claims:
        assert hit.citations, "没有引用的断言不允许出现在结果里"
        citation = hit.citations[0]
        assert citation.page >= 1
        assert citation.document_title
        row = conn.execute(
            "SELECT text, page_start, page_end FROM chunks WHERE id = %s", (citation.chunk_id,)
        ).fetchone()
        assert row is not None
        assert normalize_for_match(citation.quote) in normalize_for_match(row[0])
        assert row[1] <= citation.page <= row[2]


def test_trace_records_every_stage(seeded) -> None:
    conn, version_id = seeded
    result = SqlRetriever(conn, rerank=False).retrieve(RetrievalQuery(text="RAG training", filters=QueryFilters(version_ids=[version_id])))
    stages = [step.stage for step in result.trace]
    assert stages[:4] == ["lexical", "dense", "map_entities", "expand"]
    assert "budget" in stages


def test_empty_query_returns_explicit_empty(seeded) -> None:
    conn, version_id = seeded
    result = SqlRetriever(conn, rerank=False).retrieve(
        RetrievalQuery(text="zzzqqqxxx", filters=QueryFilters(version_ids=[version_id]))
    )
    assert result.status == "empty"
    assert result.claims == ()
    assert result.warnings


def test_missing_dense_channel_is_disclosed(seeded) -> None:
    conn, version_id = seeded
    result = SqlRetriever(conn, rerank=False).retrieve(
        RetrievalQuery(text="RAG improves accuracy", mode="hybrid", filters=QueryFilters(version_ids=[version_id]))
    )
    assert any("向量通道未接入" in warning for warning in result.warnings)


def test_result_is_json_serializable(seeded) -> None:
    conn, _ = seeded
    result = SqlRetriever(conn, rerank=False).retrieve(RetrievalQuery(text="RAG", max_chunks=3))
    payload = result.model_dump_json()
    assert '"status"' in payload and '"claims"' in payload


def test_budget_is_respected(seeded) -> None:
    conn, _ = seeded
    result = SqlRetriever(conn, rerank=False).retrieve(RetrievalQuery(text="RAG training frozen", max_tokens=500, max_chunks=10))
    assert result.stats.tokens_used <= 500 + 200  # 首块允许一次性超出
    if result.stats.truncated:
        assert any("预算" in warning for warning in result.warnings)


def test_expansion_does_not_duplicate_claims(seeded) -> None:
    conn, _ = seeded
    result = SqlRetriever(conn, rerank=False).retrieve(RetrievalQuery(text="RAG training", expand_hops=1, max_chunks=10))
    ids = [hit.claim.id for hit in result.claims]
    assert len(ids) == len(set(ids))


def test_hops_disabled_keeps_only_direct_hits(seeded) -> None:
    conn, _ = seeded
    result = SqlRetriever(conn, rerank=False).retrieve(RetrievalQuery(text="RAG training", expand_hops=0, max_chunks=10))
    assert all(hit.hop == 0 for hit in result.claims)

def test_chunks_survive_when_no_claims_extracted(db, tmp_path) -> None:
    """段落检索到了、断言还没抽 → 返回原文 + 提示，而不是空结果。"""
    path = tmp_path / "raw.md"
    path.write_text("# 纯文本\n\n残差连接让梯度能绕过若干层直接回传。\n", encoding="utf-8")
    with db.transaction(force_rollback=True):
        version_id = load_ingest(db, ingest(path)).version_id
        result = SqlRetriever(db, rerank=False).retrieve(
            RetrievalQuery(text="残差连接", max_chunks=5, filters=QueryFilters(version_ids=(version_id,)))
        )
        assert result.status == "ok"
        assert result.chunks, "召回到段落时不能因为没有断言就返回空"
        assert result.claims == ()
        assert any("没有已抽取的断言" in w for w in result.warnings)


def test_rerank_only_reorders_head(seeded):
    """重排只动头部 RERANK_HEAD 个，尾部顺序保持不变。"""

    class ReverseReranker:
        model = "fake"

        def rerank(self, query, documents):
            return [(chunk_id, 1.0 - index * 0.1) for index, (chunk_id, _) in enumerate(reversed(documents))]

    db, _ = seeded
    plain = SqlRetriever(db, rerank=False).retrieve(RetrievalQuery(text="RAG training", max_chunks=10))
    reranked = SqlRetriever(db, rerank=True, reranker=ReverseReranker()).retrieve(
        RetrievalQuery(text="RAG training", max_chunks=10)
    )
    plain_ids = [c.chunk_id for c in plain.chunks]
    new_ids = [c.chunk_id for c in reranked.chunks]
    head = RERANK_HEAD
    assert new_ids[:head] != plain_ids[:head]  # 头部被重排
    assert new_ids[head:] == [c for c in plain_ids if c not in set(new_ids[:head])][: len(new_ids[head:])]
