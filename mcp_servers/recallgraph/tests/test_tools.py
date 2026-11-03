"""工具层测试：模型能拿到“可核对”的东西，且能拿结果里的 id 继续往下调。

三个工具的共同契约：
- 结果里的 id（chunk_id / claim_id / entity_id）都是真的、能接着用；
- 原文只从库里取，不加工；
- 找不到就说找不到，不编。
"""

from __future__ import annotations

import psycopg
import pytest

from recallgraph.extract import get_extractor
from recallgraph.extract.writer import extract_version
from recallgraph.ingest.pipeline import ingest
from recallgraph.storage import connect, ensure_schema, load_ingest, schema_ready
from recallgraph.tools import dispatch, query_knowledge_graph, read_evidence, search_materials, tool_specs

BODY = """# Retrieval Notes

## Training

RAG 的训练是端到端微调，文档编码器和查询编码器保持冻结。
The retriever has no independent supervision signal in end-to-end training.
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
    path = tmp_path / "notes.md"
    path.write_text(BODY, encoding="utf-8")
    with db.transaction(force_rollback=True):
        version_id = load_ingest(db, ingest(path)).version_id
        extract_version(db, version_id, get_extractor("stub"))
        yield db, version_id


def test_specs_are_openai_compatible() -> None:
    specs = tool_specs()
    assert {spec["function"]["name"] for spec in specs} == {
        "search_materials",
        "read_evidence",
        "query_knowledge_graph",
        "recall_memories",
        "propose_memory_update",
    }
    for spec in specs:
        function = spec["function"]
        assert spec["type"] == "function"
        assert function["description"]
        assert function["parameters"]["type"] == "object"
    search = next(s for s in specs if s["function"]["name"] == "search_materials")
    assert search["function"]["parameters"]["required"] == ["query"]


def test_search_returns_claims_whose_chunk_ids_can_be_read(seeded) -> None:
    conn, _ = seeded
    result, note = search_materials(conn, query="retriever supervision", top_k=5)

    assert note["tool"] == "search_materials"
    assert result["claims"], "应命中至少一条断言"
    for claim in result["claims"]:
        assert claim["chunk_id"], "每条断言都要能回指到段落"
        assert claim["quote"], "每条断言都要带原文引文"
        assert claim["page"] >= 1

    chunk_id = result["claims"][0]["chunk_id"]
    read_back, _ = read_evidence(conn, chunk_ids=[chunk_id])
    assert read_back["chunks"][0]["chunk_id"] == chunk_id
    original = conn.execute("SELECT text FROM chunks WHERE id = %s", (chunk_id,)).fetchone()[0]
    assert read_back["chunks"][0]["text"] in original, "回读的必须是库里原文，不能加工"


def test_read_evidence_respects_budget(seeded) -> None:
    conn, _ = seeded
    found, _ = search_materials(conn, query="training", top_k=3)
    ids = [claim["chunk_id"] for claim in found["claims"]]
    result, note = read_evidence(conn, chunk_ids=ids, max_chars=600)
    assert note["chars"] <= 600
    assert result["chunks"], "至少返回一块"


def test_read_evidence_by_document_page(seeded) -> None:
    conn, _ = seeded
    found, _ = search_materials(conn, query="training", top_k=1)
    # 用 chunk 反查文档，避免依赖内部 id 规则
    row = conn.execute(
        "SELECT v.document_id FROM chunks ch JOIN document_versions v ON v.id = ch.version_id WHERE ch.id = %s",
        (found["claims"][0]["chunk_id"],),
    ).fetchone()
    document_id = row[0]
    result, _ = read_evidence(conn, document_id=document_id, page=1, page_end=1)
    assert result["chunks"]
    assert all(chunk["document_id"] == document_id for chunk in result["chunks"])


def test_graph_returns_claims_with_evidence(seeded) -> None:
    conn, _ = seeded
    result, note = query_knowledge_graph(conn, entity="RAG", hops=1, limit=10)
    if not result["matched"]:
        pytest.skip("这篇笔记没有抽到 RAG 实体")
    assert note["tool"] == "query_knowledge_graph"
    assert result["claims"], "一跳至少有一条断言"
    for claim in result["claims"]:
        assert claim["evidence"], "图里的断言也必须带证据"


def test_graph_unknown_entity_is_explicit(seeded) -> None:
    conn, _ = seeded
    result, _ = query_knowledge_graph(conn, entity="完全不存在的实体xyz")
    assert result["matched"] == []
    assert result["claims"] == []
    assert "search_materials" in result["hint"], "失败时要告诉模型下一步该干什么"


def test_memory_tools_round_trip(seeded) -> None:
    conn, _ = seeded
    proposal, note = dispatch(
        conn,
        "propose_memory_update",
        {"key": "answer_style", "value": "先给结论", "reason": "用户要求"},
    )
    assert note["operation"] in {"add", "update", "noop"}
    assert proposal["kind"] == "memory_proposal"
    assert "尚未写入" in proposal["hint"] or "相同" in proposal["hint"]

    recalled, _ = dispatch(conn, "recall_memories", {"query": "answer_style"})
    assert isinstance(recalled["memories"], list)


def test_dispatch_rejects_unknown_tool(seeded) -> None:
    conn, _ = seeded
    with pytest.raises(ValueError):
        dispatch(conn, "delete_everything", {})