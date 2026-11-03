"""给模型用的工具：检索、读原文、图查询。

这一层只负责“取”，不负责“答”。判断该调哪个工具、调几次、要不要继续查、最后怎么组织答案，
都是模型自己的事——这里不编排工作流，只把每一步做得可组合：

- 每个结果都带 id（chunk_id / claim_id / entity_id），模型可以拿它继续调下一个工具；
- 每个结果都带出处（文档、页码、章节、原文引文），模型引用时不需要再猜；
- 工具描述写清“这是索引不是答案”，模型才会在引用前回来读原文。
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import psycopg

from recallgraph import memory
from recallgraph.dense import make_embedder
from recallgraph.retrieval import RetrievalQuery
from recallgraph.retriever import SqlRetriever

MAX_READ_CHARS = 12000


def tool_specs() -> list[dict[str, Any]]:
    """OpenAI 兼容的 function 定义。描述就是给模型的说明书。"""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_materials",
                "description": (
                    "在个人知识库里检索。返回命中的断言（带着原文引文、页码、所属段落 id）和段落预览。"
                    "这是索引不是原文：它告诉你“东西在哪、大概是什么关系”，引用前请用 read_evidence 读原文核对。"
                    "适合回答“有没有、在哪里、大致什么说法”。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "检索词，用用户问题里的关键概念，不要整句照抄"},
                        "top_k": {"type": "integer", "description": "最多返回几条断言，默认 8", "default": 8},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_evidence",
                "description": (
                    "读原文。按 chunk_id（来自 search_materials 或 query_knowledge_graph 的结果）"
                    "或按文档 + 页码范围取原文全文。引用、核对细节、看上下文时用这个。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chunk_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "要读的段落 id 列表（字符串）；与 document_id 二选一",
                        },
                        "document_id": {"type": "string", "description": "按文档读，需配合 page / page_end"},
                        "page": {"type": "integer", "description": "起始页码（从 1 开始）"},
                        "page_end": {"type": "integer", "description": "结束页码，默认与 page 相同"},
                        "max_chars": {"type": "integer", "description": "总字符上限，默认 6000", "default": 6000},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recall_memories",
                "description": (
                    "查你（用户）自己的记忆：偏好、做过的决定、改过的主意、纠正过的说法。"
                    "这些不在论文里。用户问“我之前说过什么”“你还记得我改主意了吗”"
                    "或者问题涉及他的偏好/历史决定时调用。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "关键词；留空则返回最近的几条"},
                        "as_of": {"type": "string", "description": "可选，RFC3339 时间；查“那天”用户当时的看法"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "propose_memory_update",
                "description": (
                    "提议写一条记忆（不会立刻生效，需要用户确认）。"
                    "只在用户明确表达“以后都这样”“我改主意了”“记住这一点”时调用；一次只提一条。"
                    "不要把论文内容写进记忆——那是知识库的事。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "稳定的短标识，如 answer_style、project_focus"},
                        "value": {"type": "string", "description": "一句话，人可读"},
                        "reason": {"type": "string", "description": "为什么提议写它（引用用户原话最好）"},
                    },
                    "required": ["key", "value"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_knowledge_graph",
                "description": (
                    "从某个实体出发，沿知识图谱多跳查关系。返回经过的断言和邻居实体。"
                    "适合“X 和 Y 什么关系”“X 影响了哪些工作”“顺着这条线还能找到什么”这类问题。"
                    "hops=1 是一跳邻居，hops=2 是两跳；relation 可限定关系类型。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string", "description": "实体名，例如 RAG、ResNet、残差连接"},
                        "relation": {
                            "type": "string",
                            "description": "可选，限定关系类型，例如 improves / uses / part_of / is_a",
                        },
                        "hops": {"type": "integer", "description": "跳数，默认 1，最大 3", "default": 1},
                        "limit": {"type": "integer", "description": "最多返回多少条断言，默认 20", "default": 20},
                    },
                    "required": ["entity"],
                },
            },
        },
    ]


def dispatch(
    conn: psycopg.Connection,
    name: str,
    arguments: dict[str, Any],
    *,
    retriever: SqlRetriever | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """执行一次工具调用，返回 (给模型的结果, 给日志的 trace)。"""
    if name == "search_materials":
        return search_materials(conn, retriever=retriever, **arguments)
    if name == "read_evidence":
        return read_evidence(conn, **arguments)
    if name == "query_knowledge_graph":
        return query_knowledge_graph(conn, **arguments)
    if name == "recall_memories":
        return recall_memories(conn, **arguments)
    if name == "propose_memory_update":
        return propose_memory_update(conn, **arguments)
    raise ValueError(f"未知工具：{name}")


def search_materials(
    conn: psycopg.Connection,
    *,
    query: str,
    top_k: int = 8,
    retriever: SqlRetriever | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    engine = retriever or SqlRetriever(conn)
    result = engine.retrieve(RetrievalQuery(text=query, top_k=max(1, min(int(top_k), 20))))
    claims = []
    for hit in result.claims:
        citation = hit.citations[0]
        claims.append(
            {
                "claim_id": hit.claim.id,
                "claim": hit.claim.claim_text,
                "predicate": hit.claim.predicate.value,
                "title": citation.document_title,
                "page": citation.page,
                "chunk_id": citation.chunk_id,
                "quote": citation.quote,
                "via": list(hit.channels),
            }
        )
    chunks = [
        {
            "chunk_id": chunk.chunk_id,
            "page": chunk.page,
            "section": " / ".join(chunk.section_path),
            "preview": chunk.text,
        }
        for chunk in result.chunks
    ]
    covered = {claim["chunk_id"] for claim in claims}
    uncovered = [chunk for chunk in chunks if chunk["chunk_id"] not in covered]
    payload = {
        "query": query,
        "claims": claims,
        "chunks": chunks,
        "coverage": {"claims": len(claims), "chunks": len(chunks), "uncovered_chunks": len(uncovered)},
        "warnings": result.warnings,
        "hint": (
            "claim 只是索引。uncovered_chunks>0 说明有命中段落没有被任何断言覆盖："
            "此时不能因为 claims 为空就回答“知识库里没有”，请用 read_evidence 读这些段落。"
        ),
    }
    trace = {
        "tool": "search_materials",
        "claims": len(claims),
        "chunks": len(chunks),
        "ms": round(result.stats.elapsed_ms),
    }
    return payload, trace


def read_evidence(
    conn: psycopg.Connection,
    *,
    chunk_ids: list[str] | None = None,
    document_id: str | None = None,
    page: int | None = None,
    page_end: int | None = None,
    max_chars: int = 6000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    max_chars = max(500, min(int(max_chars), MAX_READ_CHARS))
    sql = """
        SELECT ch.id, ch.version_id, v.document_id, d.title, ch.page_start, ch.page_end,
               ch.section_path, ch.text
        FROM chunks ch
        JOIN document_versions v ON v.id = ch.version_id
        JOIN documents d ON d.id = v.document_id
        WHERE {where}
        ORDER BY ch.page_start, ch.seq
        LIMIT 40
    """
    params: tuple[Any, ...]
    if chunk_ids:
        where = "ch.id = ANY(%s)"
        params = (list(chunk_ids),)
    elif document_id is not None:
        where = "v.document_id = %s AND ch.page_start >= %s AND ch.page_end <= %s"
        params = (document_id, page or 1, page_end or page or 10**6)
    else:
        raise ValueError("read_evidence 需要 chunk_ids 或 document_id")

    rows = conn.execute(sql.format(where=where), params).fetchall()
    chunks: list[dict[str, Any]] = []
    used = 0
    for chunk_id, version_id, doc_id, title, start_page, end_page, section, text in rows:
        remaining = max_chars - used
        if remaining <= 0:
            break
        body = text[:remaining]
        chunks.append(
            {
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "title": title,
                "version_id": version_id,
                "page": f"{start_page}-{end_page}",
                "section": section,
                "text": body,
                "truncated": len(body) < len(text),
            }
        )
        used += len(body)

    payload = {
        "chunks": chunks,
        "truncated": used >= max_chars,
        "hint": "这些是原文，可直接引用；引用时给出 title 与 page",
    }
    trace = {"tool": "read_evidence", "chunks": len(chunks), "chars": used}
    return payload, trace


def query_knowledge_graph(
    conn: psycopg.Connection,
    *,
    entity: str,
    relation: str | None = None,
    hops: int = 1,
    limit: int = 20,
) -> tuple[dict[str, Any], dict[str, Any]]:
    hops = max(1, min(int(hops), 3))
    limit = max(1, min(int(limit), 50))
    matched = _resolve_entity(conn, entity)
    if not matched:
        return (
            {
                "entity": entity,
                "matched": [],
                "claims": [],
                "hint": "图里没有这个实体；先用 search_materials 找准确的叫法",
            },
            {"tool": "query_knowledge_graph", "matched": 0},
        )

    entity_ids = [row[0] for row in matched]
    seen_claims: dict[str, dict[str, Any]] = {}
    seen_entities = set(entity_ids)
    frontier = list(entity_ids)
    edges: list[dict[str, Any]] = []

    for hop in range(1, hops + 1):
        claims = _claims_for(conn, frontier, relation, limit=limit)
        next_frontier: list[str] = []
        for claim in claims:
            if claim["claim_id"] in seen_claims:
                continue
            if len(seen_claims) >= limit:
                break
            seen_claims[claim["claim_id"]] = claim
            for other in (claim["subject_id"], claim["object_id"]):
                if other not in seen_entities:
                    seen_entities.add(other)
                    next_frontier.append(other)
                    edges.append(
                        {"from": claim["subject"], "relation": claim["predicate"], "to": claim["object"], "hop": hop}
                    )
        if not next_frontier:
            break
        frontier = next_frontier

    claim_rows = list(seen_claims.values())
    _attach_evidence(conn, claim_rows)
    labels = {row[0]: (row[1], row[2]) for row in _entities_by_id(conn, list(seen_entities))}

    payload = {
        "entity": entity,
        "matched": [{"entity_id": row[0], "name": row[1], "kind": row[2]} for row in matched],
        "claims": claim_rows,
        "neighbors": [
            {"entity_id": eid, "name": labels[eid][0], "kind": labels[eid][1]}
            for eid in seen_entities
            if eid not in entity_ids and eid in labels
        ],
        "edges": edges[:limit],
        "hops": hops,
        "hint": "要读某条断言的原文，用 evidence 里的 chunk_id 调 read_evidence",
    }
    trace = {"tool": "query_knowledge_graph", "matched": len(matched), "claims": len(claim_rows)}
    return payload, trace


def recall_memories(
    conn: psycopg.Connection,
    *,
    query: str | None = None,
    as_of: str | None = None,
    retriever: SqlRetriever | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    moment = None
    if as_of:
        moment = datetime.fromisoformat(as_of)
    embedder = None
    if os.environ.get("DASHSCOPE_API_KEY"):
        try:
            embedder = make_embedder(None)
        except (RuntimeError, OSError):
            embedder = None
    items = memory.recall(conn, query, as_of=moment, embedder=embedder)

    # 记忆不是结论：把与它最相关的知识库断言一并给模型，它才有机会反驳用户。
    related: list[dict[str, Any]] = []
    if items and moment is None:
        engine = retriever or SqlRetriever(conn)
        probe = engine.retrieve(RetrievalQuery(text=items[0]["value"], top_k=3))
        related = [
            {
                "claim": hit.claim.claim_text,
                "title": hit.citations[0].document_title,
                "page": hit.citations[0].page,
                "chunk_id": hit.citations[0].chunk_id,
            }
            for hit in probe.claims[:3]
        ]
    payload = {
        "memories": [
            {
                "key": item["key"],
                "value": item["value"],
                "valid_from": item["valid_from"].isoformat(),
                "recorded_at": item["recorded_at"].isoformat(),
                "status": item["status"],
            }
            for item in items
        ],
        "related_evidence": related,
        "as_of": as_of,
        "hint": (
            "记忆是用户自己的说法，不是论文结论。related_evidence 是知识库里与这条记忆最接近的断言："
            "如果两者矛盾，以证据为准，并明确告诉用户哪句话对不上、依据是哪篇第几页。"
        ),
    }
    return payload, {"tool": "recall_memories", "memories": len(items), "related": len(related)}


def propose_memory_update(
    conn: psycopg.Connection, *, key: str, value: str, reason: str = ""
) -> tuple[dict[str, Any], dict[str, Any]]:
    proposal = memory.propose(conn, key=key, value=value, reason=reason)
    return proposal, {"tool": "propose_memory_update", "operation": proposal["operation"], "key": proposal["key"]}


def _resolve_entity(conn: psycopg.Connection, name: str) -> list[tuple[str, str, str]]:
    rows = conn.execute(
        "SELECT id, canonical_name, kind::text FROM entities WHERE lower(canonical_name) = lower(%s) LIMIT 5", (name,)
    ).fetchall()
    if rows:
        return rows
    return conn.execute(
        "SELECT id, canonical_name, kind::text FROM entities WHERE canonical_name ILIKE %s ORDER BY length(canonical_name) LIMIT 5",
        (f"%{name}%",),
    ).fetchall()


def _claims_for(
    conn: psycopg.Connection, entity_ids: list[str], relation: str | None, *, limit: int
) -> list[dict[str, Any]]:
    sql = """
        SELECT c.id, c.subject_id, s.canonical_name, c.predicate::text, o.canonical_name, c.object_id, c.claim_text
        FROM claims c
        JOIN entities s ON s.id = c.subject_id
        JOIN entities o ON o.id = c.object_id
        WHERE c.status = 'active'
          AND (c.subject_id = ANY(%s) OR c.object_id = ANY(%s))
    """
    params: list[Any] = [entity_ids, entity_ids]
    if relation:
        sql += " AND c.predicate = %s"
        params.append(relation)
    sql += " ORDER BY c.id LIMIT %s"
    params.append(limit)
    return [
        {
            "claim_id": row[0],
            "subject_id": row[1],
            "subject": row[2],
            "predicate": row[3],
            "object": row[4],
            "object_id": row[5],
            "claim": row[6],
        }
        for row in conn.execute(sql, params).fetchall()
    ]


def _attach_evidence(conn: psycopg.Connection, claims: list[dict[str, Any]]) -> None:
    if not claims:
        return
    ids = [claim["claim_id"] for claim in claims]
    rows = conn.execute(
        """
        SELECT e.claim_id, e.chunk_id, e.quote, ch.page_start, ch.page_end, d.title
        FROM evidence e
        JOIN chunks ch ON ch.id = e.chunk_id
        JOIN document_versions v ON v.id = ch.version_id
        JOIN documents d ON d.id = v.document_id
        WHERE e.claim_id = ANY(%s)
        """,
        (ids,),
    ).fetchall()
    by_claim: dict[str, list[dict[str, Any]]] = {}
    for claim_id, chunk_id, quote, page_start, page_end, title in rows:
        by_claim.setdefault(claim_id, []).append(
            {"chunk_id": chunk_id, "quote": quote, "page": f"{page_start}-{page_end}", "title": title}
        )
    for claim in claims:
        claim["evidence"] = by_claim.get(claim["claim_id"], [])


def _entities_by_id(conn: psycopg.Connection, ids: list[str]) -> list[tuple[str, str, str]]:
    if not ids:
        return []
    return conn.execute("SELECT id, canonical_name, kind::text FROM entities WHERE id = ANY(%s)", (ids,)).fetchall()
