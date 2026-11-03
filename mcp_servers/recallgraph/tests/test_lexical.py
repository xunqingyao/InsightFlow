"""关键词通道：分词、倒排表、BM25。

分两类：
- 纯函数测试（不需要数据库）：分词规则、词频统计；
- 集成测试（需要 PostgreSQL）：入库后倒排表是否建好、BM25 是否把相关段落排前面。
"""

from __future__ import annotations

import psycopg
import pytest

from recallgraph.ingest.pipeline import ingest
from recallgraph.lexical import index_stats, index_version, search_lexical, term_frequencies, tokenize
from recallgraph.storage import connect, ensure_schema, load_ingest
from recallgraph.storage.db import SQL_DIR


@pytest.fixture(scope="module")
def db():
    try:
        conn = connect()
    except psycopg.OperationalError:
        pytest.skip("PostgreSQL 不可用：先执行 docker compose up -d")
    ensure_schema(conn)
    yield conn
    conn.close()


@pytest.fixture()
def tx(db):
    with db.transaction(force_rollback=True):
        yield db


def _write(tmp_path, name: str, body: str):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


# ── 分词 ───────────────────────────────────────────────────────────────────


def test_tokenize_english_lowercases_and_drops_stopwords():
    tokens = tokenize("The Retrieval Augmented Model is trained WITH gradient descent")
    assert "retriev" in tokens  # Porter 词干：retrieval → retriev
    assert "the" not in tokens
    assert "is" not in tokens
    assert "with" not in tokens


def test_tokenize_chinese_segments_by_word():
    tokens = tokenize("检索增强生成的训练方式是端到端微调")
    assert "检索" in tokens
    assert "增强" in tokens
    assert "训练" in tokens
    assert "的" not in tokens


def test_tokenize_mixed_text_keeps_both_sides():
    tokens = tokenize("RAG 用 BM25 和向量检索，BART 做生成")
    assert {"rag", "bm25", "bart"} <= set(tokens)
    assert {"检索", "生成"} <= set(tokens)


def test_tokenize_normalizes_fullwidth_and_case():
    assert tokenize("ＲＡＧ") == ["rag"]
    assert tokenize("BERT") == ["bert"]


def test_term_frequencies_counts_repeats():
    assert term_frequencies(["rag", "rag", "bm25"]) == {"rag": 2, "bm25": 1}


# ── 入库与倒排表 ───────────────────────────────────────────────────────────


def test_load_builds_inverted_index(tx, tmp_path):
    result = ingest(_write(tmp_path, "a.md", "# Index Paper\n\n## Method\n\n" + "vector index retrieval. " * 30))
    outcome = load_ingest(tx, result)

    rows = tx.execute(
        "SELECT count(*), count(DISTINCT term) FROM chunk_terms WHERE chunk_id LIKE %s",
        (outcome.version_id + ":%",),
    ).fetchone()
    assert rows[0] > 0
    assert rows[1] > 0


def test_reindex_is_idempotent(tx, tmp_path):
    result = ingest(_write(tmp_path, "b.md", "# Paper B\n\n## Intro\n\n" + "graph traversal depth. " * 30))
    outcome = load_ingest(tx, result)
    before = index_stats(tx)

    index_version(tx, outcome.version_id)
    after = index_stats(tx)

    assert after["term_rows"] == before["term_rows"]


def test_search_ranks_relevant_paper_first(tx, tmp_path):
    lexical = ingest(_write(tmp_path, "lex.md", "# Lexical Search\n\n## BM25\n\n" + "bm25 term frequency saturation ranking. " * 30))
    residual = ingest(_write(tmp_path, "res.md", "# Residual Nets\n\n## Depth\n\n" + "residual shortcut connection depth degradation. " * 30))
    load_ingest(tx, lexical)
    load_ingest(tx, residual)

    hits = search_lexical(tx, "residual connection depth", top_k=3)

    assert hits
    assert hits[0].title == "Residual Nets"
    assert hits[0].page_start >= 1
    assert hits[0].section_path == ["Residual Nets", "Depth"]


def test_search_without_known_terms_returns_empty(tx):
    assert search_lexical(tx, "zzzqqq", top_k=3) == []


def test_search_can_be_restricted_to_one_version(tx, tmp_path):
    paper = ingest(_write(tmp_path, "c.md", "# Reranking\n\n## Method\n\n" + "rerank candidate passages. " * 30))
    other = ingest(_write(tmp_path, "d.md", "# Other\n\n## Method\n\n" + "rerank candidate passages. " * 30))
    load_ingest(tx, paper)
    load_ingest(tx, other)

    hits = search_lexical(tx, "rerank candidate passages", top_k=3, version_ids=[paper.version.id])

    assert hits
    assert {hit.version_id for hit in hits} == {paper.version.id}


def test_chunk_terms_reference_existing_chunks(tx, tmp_path):
    result = ingest(_write(tmp_path, "e.md", "# Ref\n\n## X\n\n" + "alpha beta gamma. " * 30))
    outcome = load_ingest(tx, result)

    orphan = tx.execute(
        "SELECT count(*) FROM chunk_terms ct LEFT JOIN chunks c ON c.id = ct.chunk_id WHERE c.id IS NULL"
    ).fetchone()[0]
    assert orphan == 0
    assert outcome.chunks_written > 0


def test_lexical_migration_recorded(db):
    done = {row[0] for row in db.execute("SELECT version FROM schema_migrations")}
    assert {p.stem for p in SQL_DIR.glob("*.sql")} <= done