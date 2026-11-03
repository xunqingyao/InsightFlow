"""存储层集成测试。

需要本机跑着 PostgreSQL（`docker compose up -d`）；连不上时整组跳过，
不阻塞不需要数据库的单元测试。
"""

from __future__ import annotations

import psycopg
import pytest

from recallgraph.ingest.pipeline import ingest
from recallgraph.storage import connect, counts, ensure_schema, load_ingest, schema_ready
from recallgraph.storage.db import SQL_DIR


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
    """每个用例一个事务，结束回滚，互不污染。"""
    with db.transaction(force_rollback=True):
        yield db


def _paper(tmp_path, name="paper.md", body="# Test Paper\n\n## Method\n\n" + "Retrieval augments generation. " * 40):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_migrations_are_idempotent(db):
    assert ensure_schema(db) == []
    versions = {p.stem for p in SQL_DIR.glob("*.sql")}
    done = {row[0] for row in db.execute("SELECT version FROM schema_migrations")}
    assert versions <= done


def test_load_writes_document_version_and_chunks(tx, tmp_path):
    result = ingest(_paper(tmp_path))
    outcome = load_ingest(tx, result)

    assert outcome.deduped is False
    assert outcome.chunks_written == len(result.chunks) > 0

    row = tx.execute(
        "SELECT document_id, page_count, parser_version FROM document_versions WHERE id = %s",
        (outcome.version_id,),
    ).fetchone()
    assert row == (outcome.document_id, result.version.page_count, result.version.parser_version)

    stored = tx.execute(
        "SELECT seq, page_start, section_path, text FROM chunks WHERE version_id = %s ORDER BY seq",
        (outcome.version_id,),
    ).fetchall()
    assert [c[0] for c in stored] == list(range(len(result.chunks)))
    assert stored[0][3] == result.chunks[0].text
    assert stored[0][2] == list(result.chunks[0].section_path)


def test_same_file_is_deduped(tx, tmp_path):
    result = ingest(_paper(tmp_path))
    first = load_ingest(tx, result)
    second = load_ingest(tx, result)

    assert second.deduped is True
    assert second.version_id == first.version_id
    assert second.chunks_written == 0
    total = tx.execute("SELECT count(*) FROM chunks WHERE version_id = %s", (first.version_id,)).fetchone()[0]
    assert total == len(result.chunks)


def test_two_versions_share_one_document(tx, tmp_path):
    v1 = ingest(_paper(tmp_path, "v1.md", "# Same Title\n\n## A\n\n" + "alpha beta gamma. " * 40))
    load_ingest(tx, v1)
    v2 = ingest(_paper(tmp_path, "v2.md", "# Same Title\n\n## A\n\n" + "alpha beta gamma delta. " * 40))
    second = load_ingest(tx, v2)

    assert second.document_id == v1.document.id
    assert second.version_id != v1.version.id


def test_chunks_are_immutable(tx, tmp_path):
    result = ingest(_paper(tmp_path))
    outcome = load_ingest(tx, result)

    with pytest.raises(psycopg.errors.RaiseException, match="不可变"), tx.transaction():
        tx.execute("UPDATE chunks SET text = 'rewritten' WHERE version_id = %s", (outcome.version_id,))

    assert counts(tx)["chunks"] > 0