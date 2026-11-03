"""MCP 服务与笔记工具：工具清单要对得上，笔记能写能读、能挂断言。"""

from __future__ import annotations

import asyncio
import uuid

import psycopg
import pytest

from recallgraph import notes
from recallgraph.mcp_server import server
from recallgraph.storage import connect, ensure_schema, schema_ready


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


def test_server_exposes_six_tools() -> None:
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert names == {
        "search_materials",
        "read_evidence",
        "query_knowledge_graph",
        "recall_memories",
        "propose_memory_update",
        "create_or_update_note",
    }
    for tool in tools:
        assert tool.description and len(tool.description) > 20
        assert tool.input_schema["type"] == "object"


def test_instructions_mention_evidence_not_answer() -> None:
    assert "证据" in (server.instructions or "")
    assert "页码" in (server.instructions or "")


def test_note_roundtrip(db) -> None:
    title = f"pytest 笔记 {uuid.uuid4().hex[:6]}"
    created = notes.create_or_update_note(db, title=title, body="这是测试写的推论。", author="pytest")
    note_id = created["note_id"]
    assert note_id.startswith("NOTE-")

    got = notes.get_note(db, note_id=note_id)
    assert got is not None
    assert got["title"] == title
    assert got["body"] == "这是测试写的推论。"
    assert got["author"] == "pytest"

    updated = notes.create_or_update_note(db, note_id=note_id, title=title, body="改过的正文。")
    assert updated["note_id"] == note_id
    assert notes.get_note(db, note_id=note_id)["body"] == "改过的正文。"


def test_note_links_claims_and_reports_missing(db) -> None:
    claim = db.execute("SELECT id FROM claims WHERE status = 'active' LIMIT 1").fetchone()
    if claim is None:
        pytest.skip("库里没有 active 断言")
    created = notes.create_or_update_note(
        db,
        title=f"pytest 挂断言 {uuid.uuid4().hex[:6]}",
        body="引用了断言。",
        claim_ids=[claim[0], "CLM-NOT-EXIST"],
    )
    assert created["linked_claims"] == [claim[0]]
    assert created["missing_claims"] == ["CLM-NOT-EXIST"]
    got = notes.get_note(db, note_id=created["note_id"])
    assert [c["claim_id"] for c in got["claims"]] == [claim[0]]


def test_note_rejects_empty(db) -> None:
    with pytest.raises(ValueError):
        notes.create_or_update_note(db, title="", body="x")
    with pytest.raises(ValueError):
        notes.create_or_update_note(db, title="x", body="   ")
    with pytest.raises(LookupError):
        notes.create_or_update_note(db, note_id="NOTE-NOPE", title="x", body="y")