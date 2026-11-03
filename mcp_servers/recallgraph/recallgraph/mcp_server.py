"""RecallGraph 的 MCP 服务（stdio）。

把"知识库"包成一个 MCP 服务器，让任何支持 MCP 的宿主（DeerFlow/InsightFlow、
Claude Desktop、Cursor……）把它当工具用。服务本身**不编排流程**：它只提供
取数能力，什么时候搜、搜几轮、要不要展开图、要不要读原文，由宿主的模型决定。

运行：

    RECALLGRAPH_DSN=postgresql://recallgraph:recallgraph@127.0.0.1:5434/recallgraph \
    .venv/bin/python -m recallgraph.mcp_server

返回值统一信封，与 InsightFlow 的 mcp_servers 一致：

    {"ok": true,  "data": {...}, "error": null, "metadata": {...}}
    {"ok": false, "data": null,  "error": {"code": "...", "message": "..."}, "metadata": {...}}
"""

from __future__ import annotations

import os
from typing import Any

import psycopg
from mcp.server.mcpserver import MCPServer

from recallgraph import notes, tools

INSTRUCTIONS = """\
RecallGraph 是一个论文与个人笔记的知识库，返回的是**证据**，不是结论。

调用建议：
1. 先用 search_materials 找相关断言与段落，返回里每条都带 claim_id / chunk_id / 文档 / 页码 / 引文；
2. 断言是索引，会丢失条件与上下文。涉及"为什么""怎么定义""具体数字"时，
   用 read_evidence 读原文（chunk_ids=...）再下结论；
3. 想看两个概念之间的关系、或从一条线索往外扩，用 query_knowledge_graph（hops 控制跳数）；
4. 用户提到自己过去的偏好或结论时，先 recall_memories；它返回的记忆带 valid_from / recorded_at，
   可以看到"当时怎么想"和"后来改没改"；
5. 如果记忆与论文证据冲突，**以证据为准并指出冲突**，不要顺着用户的说法编；
6. 回答里每个事实性结论都要给出来源（文档标题 + 页码），没有证据就明说没有；
7. 需要记录自己的推论时用 create_or_update_note，不要把它写成断言。
"""

server = MCPServer(name="recallgraph", version="0.1.0", instructions=INSTRUCTIONS)

DEFAULT_DSN = "postgresql://recallgraph:recallgraph@127.0.0.1:5434/recallgraph"

_conn: psycopg.Connection | None = None


def _db() -> psycopg.Connection:
    """一个进程一条连接；MCP stdio 是单客户端长连接，够用。"""
    global _conn
    dsn = os.environ.get("RECALLGRAPH_DSN", DEFAULT_DSN)
    if _conn is None or _conn.closed:
        _conn = psycopg.connect(dsn, autocommit=True, connect_timeout=5)
    return _conn


def _ok(data: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"ok": True, "data": data, "error": None, "metadata": metadata or {}}


def _fail(code: str, message: str, **metadata: Any) -> dict[str, Any]:
    return {"ok": False, "data": None, "error": {"code": code, "message": message}, "metadata": metadata}


@server.tool(
    name="search_materials",
    description=(
        "在论文与笔记库里检索，返回证据包：断言 + 原文引文 + 文档标题 + 页码，"
        "附带通道（关键词/向量/重排）与是否存在冲突。这是第一级检索，"
        "结果只是索引，需要细节时用 read_evidence 读原文。"
    ),
)
def search_materials(query: str, top_k: int = 8) -> dict[str, Any]:
    try:
        result, trace = tools.dispatch(_db(), "search_materials", {"query": query, "top_k": top_k})
    except Exception as exc:  # noqa: BLE001 - 工具边界要兜住
        return _fail("search_failed", str(exc))
    return _ok(result, {"tool": "search_materials", **trace})


@server.tool(
    name="read_evidence",
    description=(
        "按 chunk_id 读取原文（可多段），或按文档/页码读取。用于核对引文、"
        "补充条件与上下文。一次最多读 12000 字符。"
    ),
)
def read_evidence(
    chunk_ids: list[str] | None = None,
    document_id: str | None = None,
    page: int | None = None,
    page_end: int | None = None,
    max_chars: int = 6000,
) -> dict[str, Any]:
    try:
        result, trace = tools.dispatch(
            _db(),
            "read_evidence",
            {"chunk_ids": chunk_ids, "document_id": document_id, "page": page, "page_end": page_end, "max_chars": max_chars},
        )
    except Exception as exc:  # noqa: BLE001
        return _fail("read_failed", str(exc))
    return _ok(result, {"tool": "read_evidence", **trace})


@server.tool(
    name="query_knowledge_graph",
    description=(
        "在知识图上按实体名查关系，可多跳（hops 1-3）。返回邻居实体与关系、"
        "关联断言、来源文档。用于探索概念之间的关系，不用于精确事实核对。"
    ),
)
def query_knowledge_graph(entity: str, relation: str | None = None, hops: int = 1, limit: int = 20) -> dict[str, Any]:
    try:
        result, trace = tools.dispatch(
            _db(),
            "query_knowledge_graph",
            {"entity": entity, "relation": relation, "hops": hops, "limit": limit},
        )
    except Exception as exc:  # noqa: BLE001
        return _fail("graph_failed", str(exc))
    return _ok(result, {"tool": "query_knowledge_graph", **trace})


@server.tool(
    name="recall_memories",
    description=(
        "召回用户的历史记忆（偏好、决定、纠正过的说法）。可按 query 做关键词+向量混合检索，"
        "或按 as_of 时间点回溯。返回 valid_from / recorded_at / 是否被取代。"
    ),
)
def recall_memories(query: str | None = None, as_of: str | None = None) -> dict[str, Any]:
    try:
        result, trace = tools.dispatch(_db(), "recall_memories", {"query": query, "as_of": as_of})
    except Exception as exc:  # noqa: BLE001
        return _fail("memory_failed", str(exc))
    return _ok(result, {"tool": "recall_memories", **trace})


@server.tool(
    name="propose_memory_update",
    description=(
        "提议写入/更新一条记忆。只生成提案，不落库；需要调用方确认后再 commit_memory。"
        "用于用户明确表达了偏好或纠正时。"
    ),
)
def propose_memory_update(key: str, value: str, reason: str = "") -> dict[str, Any]:
    try:
        result, trace = tools.dispatch(_db(), "propose_memory_update", {"key": key, "value": value, "reason": reason})
    except Exception as exc:  # noqa: BLE001
        return _fail("propose_failed", str(exc))
    return _ok(result, {"tool": "propose_memory_update", **trace})


@server.tool(
    name="create_or_update_note",
    description=(
        "新建或更新一条笔记（人的推测、总结、待办），可挂到若干断言上。"
        "笔记不是论文结论，写的时候要说清是推论。"
    ),
)
def create_or_update_note(
    title: str,
    body: str,
    note_id: str | None = None,
    claim_ids: list[str] | None = None,
) -> dict[str, Any]:
    try:
        result = notes.create_or_update_note(
            _db(),
            title=title,
            body=body,
            note_id=note_id,
            claim_ids=claim_ids or [],
        )
    except Exception as exc:  # noqa: BLE001
        return _fail("note_failed", str(exc))
    return _ok(result, {"tool": "create_or_update_note"})


def main() -> None:
    try:
        server.run(transport="stdio")
    finally:
        if _conn is not None and not _conn.closed:
            _conn.close()


if __name__ == "__main__":
    main()