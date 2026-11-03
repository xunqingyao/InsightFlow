"""记忆层：关于“你”的事实，不是关于论文的事实。

三条设计：

1. **key 稳定**：一条记忆 = (namespace, key) 的当前值；value 是一句话，人可读。
2. **只追加**：改记忆不是改行，是新增一条带 supersedes_id 的事件，旧事件 status 置 superseded。
   旧值的“有效期终点”不写进 valid_to——这张表只允许改 status，终点由下一条事件的 valid_from 推导。
3. **写前先提议**：模型只能 propose，落库要 commit。它理解错一次，最多提个错建议，改不了历史。

时间有两种，不能混：
- valid_from / valid_to：这条事实在现实里什么时候成立；
- recorded_at：系统什么时候知道的。
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import psycopg

from recallgraph.dense import _vector_literal, make_embedder
from recallgraph.ids import new_ulid
from recallgraph.schema import utcnow

DEFAULT_USER = "me"
DEFAULT_NAMESPACE = "user"
MAX_VALUE_CHARS = 500
_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")

_COLUMNS = "id, memory_key, value, valid_from, valid_to, recorded_at, status, supersedes_id"
_COLUMNS_E = ", ".join(f"e.{column.strip()}" for column in _COLUMNS.split(","))


class MemoryError(ValueError):
    """记忆操作的输入不合法。"""


def normalize_key(key: str) -> str:
    candidate = key.strip().lower()
    if not _KEY_RE.match(candidate):
        raise MemoryError(f"key 只能是小写字母、数字、下划线、点和短横线（≤64 字符）：{key!r}")
    return candidate


def normalize_value(value: str) -> str:
    text = " ".join(value.split())
    if not text:
        raise MemoryError("value 不能为空")
    if len(text) > MAX_VALUE_CHARS:
        raise MemoryError(f"value 太长（{len(text)} > {MAX_VALUE_CHARS}），记忆应当是一句话")
    return text


def current(
    conn: psycopg.Connection, key: str, *, user: str = DEFAULT_USER, namespace: str = DEFAULT_NAMESPACE
) -> dict[str, Any] | None:
    """某个 key 的当前值；没有就返回 None。"""
    row = conn.execute(
        f"""
        SELECT {_COLUMNS}
        FROM memory_events
        WHERE user_id = %s AND namespace = %s AND memory_key = %s AND status = 'active'
        LIMIT 1
        """,
        (user, namespace, normalize_key(key)),
    ).fetchone()
    return _row(row, namespace=namespace) if row else None


def memory_text(key: str, value: str) -> str:
    """编码成向量时用的文本：key 也进去，否则“回答风格”这类词在 value 里不出现。"""
    return f"{key}｜{value}" if value else key


def index_memory(conn: psycopg.Connection, event_id: str, *, model: str | None = None, embedder: Any = None) -> int:
    """给一条记忆事件建向量；已存在则覆盖。返回写入行数。"""
    row = conn.execute("SELECT memory_key, value FROM memory_events WHERE id = %s", (event_id,)).fetchone()
    if row is None:
        return 0
    engine = embedder or make_embedder(model)
    vector = engine.embed([memory_text(row[0], row[1])])[0]
    conn.execute(
        """
        INSERT INTO memory_vectors (memory_event_id, model, embedding)
        VALUES (%s, %s, %s::vector)
        ON CONFLICT (memory_event_id) DO UPDATE SET model = EXCLUDED.model, embedding = EXCLUDED.embedding
        """,
        (event_id, engine.model_name, _vector_literal(vector)),
    )
    conn.commit()
    return 1


def index_all(
    conn: psycopg.Connection, *, user: str = DEFAULT_USER, namespace: str = DEFAULT_NAMESPACE, embedder: Any = None, force: bool = False
) -> int:
    """把还没建向量的记忆补齐。force=True 时全部重建。"""
    engine = embedder or make_embedder(None)
    sql = """
        SELECT e.id, e.memory_key, e.value
        FROM memory_events e
        LEFT JOIN memory_vectors v ON v.memory_event_id = e.id AND v.model = %s
        WHERE e.user_id = %s AND e.namespace = %s {extra}
        ORDER BY e.recorded_at
    """
    extra = "" if force else "AND v.memory_event_id IS NULL"
    rows = conn.execute(sql.format(extra=extra), (engine.model_name, user, namespace)).fetchall()
    if not rows:
        return 0
    vectors = engine.embed([memory_text(row[1], row[2]) for row in rows])
    for row, vector in zip(rows, vectors, strict=True):
        conn.execute(
            """
            INSERT INTO memory_vectors (memory_event_id, model, embedding)
            VALUES (%s, %s, %s::vector)
            ON CONFLICT (memory_event_id) DO UPDATE SET model = EXCLUDED.model, embedding = EXCLUDED.embedding
            """,
            (row[0], engine.model_name, _vector_literal(vector)),
        )
    conn.commit()
    return len(rows)


def recall(
    conn: psycopg.Connection,
    query: str | None = None,
    *,
    user: str = DEFAULT_USER,
    namespace: str = DEFAULT_NAMESPACE,
    as_of: datetime | None = None,
    limit: int = 10,
    embedder: Any = None,
) -> list[dict[str, Any]]:
    """找相关记忆：关键词 + 向量（可选）做 RRF 融合。

    关键词能命中“原词”，向量能命中“同义说法”（改主意 / 调整方向）。
    默认只返回“现在还有效”的；传 as_of 就回到那一天。
    """
    limit = max(1, min(int(limit), 50))
    tokens = [token for token in re.split(r"[\s,，、;；]+", (query or "").strip()) if token]

    lexical: list[dict[str, Any]] = []
    if tokens:
        where = " OR ".join(["memory_key ILIKE %s OR value ILIKE %s"] * len(tokens))
        params: list[Any] = []
        for token in tokens:
            params.extend([f"%{token}%", f"%{token}%"])
        rows = conn.execute(
            f"""
            SELECT {_COLUMNS}
            FROM memory_events
            WHERE user_id = %s AND namespace = %s AND ({where})
            ORDER BY recorded_at DESC, id DESC
            LIMIT %s
            """,
            (user, namespace, *params, limit * 5),
        ).fetchall()
        lexical = [_row(row, namespace=namespace) for row in rows]
        # 命中的词越多越靠前：避免“重点”这种泛词把无关条目顶上来。
        lexical.sort(
            key=lambda item: (
                -sum(1 for token in tokens if token in item["key"] or token in item["value"]),
                -item["recorded_at"].timestamp(),
            )
        )

    semantic: list[dict[str, Any]] = []
    if embedder is not None and (query or "").strip():
        vector = _vector_literal(embedder.embed([query.strip()])[0])
        rows = conn.execute(
            f"""
            SELECT {_COLUMNS_E}, 1 - (v.embedding <=> %s::vector) AS score
            FROM memory_vectors v
            JOIN memory_events e ON e.id = v.memory_event_id
            WHERE v.model = %s AND e.user_id = %s AND e.namespace = %s AND e.status = 'active'
            ORDER BY v.embedding <=> %s::vector
            LIMIT %s
            """,
            (vector, embedder.model_name, user, namespace, vector, limit * 2),
        ).fetchall()
        for row in rows:
            item = _row(row[:-1], namespace=namespace)
            item["score"] = float(row[-1])
            semantic.append(item)

    if semantic:
        # RRF：两路各按名次计分，同一 event 相加——同义命中与字面命中互相抬。
        scores: dict[str, float] = {}
        items_by_id: dict[str, dict[str, Any]] = {}
        for ranked in (lexical, semantic):
            for rank, item in enumerate(ranked, 1):
                items_by_id.setdefault(item["event_id"], item)
                scores[item["event_id"]] = scores.get(item["event_id"], 0.0) + 1.0 / (60 + rank)
        merged = sorted(items_by_id.values(), key=lambda item: -scores[item["event_id"]])
        merged = [dict(item, rrf=round(scores[item["event_id"]], 6)) for item in merged]
    else:
        merged = lexical

    if as_of is None:
        return [item for item in merged if item["status"] == "active"][:limit]

    # 时间旅行：终点从 supersedes 链推——谁取代了它、取代者什么时候生效。
    successors: dict[str, datetime] = {}
    for item in merged:
        parent = item["supersedes_id"]
        if parent:
            successors[parent] = min(successors.get(parent, item["valid_from"]), item["valid_from"])

    def alive(item: dict[str, Any]) -> bool:
        if item["status"] == "retracted" or item["valid_from"] > as_of:
            return False
        end = successors.get(item["event_id"])
        if end is not None and end <= as_of:
            return False
        return not (item["valid_to"] is not None and item["valid_to"] <= as_of)

    return [item for item in merged if alive(item)][:limit]


def propose(
    conn: psycopg.Connection,
    *,
    key: str,
    value: str,
    reason: str = "",
    user: str = DEFAULT_USER,
    namespace: str = DEFAULT_NAMESPACE,
) -> dict[str, Any]:
    """生成一个待确认的提案：不写库，只描述“如果确认会发生什么”。"""
    key = normalize_key(key)
    value = normalize_value(value)
    existing = current(conn, key, user=user, namespace=namespace)
    if existing and existing["value"] == value:
        operation = "noop"
    elif existing:
        operation = "update"
    else:
        operation = "add"
    if operation == "noop":
        hint = "与当前值相同，确认也不会产生新事件"
    else:
        hint = f"这是提案，尚未写入。确认后执行：python -m recallgraph.cli memory set {namespace} {key} {value!r}"
    return {
        "kind": "memory_proposal",
        "operation": operation,
        "namespace": namespace,
        "user": user,
        "key": key,
        "value": value,
        "reason": reason.strip(),
        "supersedes_id": existing["event_id"] if existing else None,
        "previous_value": existing["value"] if existing else None,
        "hint": hint,
    }


def commit(
    conn: psycopg.Connection,
    proposal: dict[str, Any],
    *,
    user: str | None = None,
    namespace: str | None = None,
) -> dict[str, Any]:
    """确认提案：add 或 update（旧事件标 superseded），同值则 noop。"""
    if proposal.get("kind") != "memory_proposal":
        raise MemoryError("commit 只接受 propose 的返回值")
    key = normalize_key(proposal["key"])
    value = normalize_value(proposal["value"])
    user = user or proposal.get("user") or DEFAULT_USER
    namespace = namespace or proposal.get("namespace") or DEFAULT_NAMESPACE

    existing = current(conn, key, user=user, namespace=namespace)
    if existing and existing["value"] == value:
        return {"status": "noop", "key": key, "value": value}

    now = utcnow()
    event_id = new_ulid()
    if existing:
        conn.execute("UPDATE memory_events SET status = 'superseded' WHERE id = %s", (existing["event_id"],))
        conn.execute(
            """
            INSERT INTO memory_events
                (id, user_id, namespace, memory_key, operation, value, valid_from, recorded_at, supersedes_id, status)
            VALUES (%s, %s, %s, %s, 'update', %s, %s, %s, %s, 'active')
            """,
            (event_id, user, namespace, key, value, now, now, existing["event_id"]),
        )
        operation = "update"
    else:
        conn.execute(
            """
            INSERT INTO memory_events
                (id, user_id, namespace, memory_key, operation, value, valid_from, recorded_at, status)
            VALUES (%s, %s, %s, %s, 'add', %s, %s, %s, 'active')
            """,
            (event_id, user, namespace, key, value, now, now),
        )
        operation = "add"
    conn.commit()
    return {"status": "committed", "operation": operation, "event_id": event_id, "key": key, "value": value}


def retract(
    conn: psycopg.Connection,
    key: str,
    *,
    reason: str = "",
    user: str = DEFAULT_USER,
    namespace: str = DEFAULT_NAMESPACE,
) -> dict[str, Any]:
    """作废当前值：不删历史，只新增一条 retract 事件。"""
    existing = current(conn, key, user=user, namespace=namespace)
    if not existing:
        return {"status": "missing", "key": normalize_key(key)}
    now = utcnow()
    event_id = new_ulid()
    conn.execute("UPDATE memory_events SET status = 'retracted' WHERE id = %s", (existing["event_id"],))
    conn.execute(
        """
        INSERT INTO memory_events
            (id, user_id, namespace, memory_key, operation, value, valid_from, recorded_at, supersedes_id, status)
        VALUES (%s, %s, %s, %s, 'retract', %s, %s, %s, %s, 'retracted')
        """,
        (event_id, user, namespace, existing["key"], reason.strip() or "retracted", now, now, existing["event_id"]),
    )
    conn.commit()
    return {"status": "retracted", "event_id": event_id, "key": existing["key"]}


def history(
    conn: psycopg.Connection, key: str, *, user: str = DEFAULT_USER, namespace: str = DEFAULT_NAMESPACE
) -> list[dict[str, Any]]:
    """一条记忆的完整历史，按时间正序。"""
    rows = conn.execute(
        f"""
        SELECT {_COLUMNS}
        FROM memory_events
        WHERE user_id = %s AND namespace = %s AND memory_key = %s
        ORDER BY recorded_at, id
        """,
        (user, namespace, normalize_key(key)),
    ).fetchall()
    return [_row(row, namespace=namespace) for row in rows]


def list_keys(
    conn: psycopg.Connection, *, user: str = DEFAULT_USER, namespace: str = DEFAULT_NAMESPACE
) -> list[dict[str, Any]]:
    """所有当前有效的记忆。"""
    rows = conn.execute(
        f"""
        SELECT {_COLUMNS}
        FROM memory_events
        WHERE user_id = %s AND namespace = %s AND status = 'active'
        ORDER BY memory_key
        """,
        (user, namespace),
    ).fetchall()
    return [_row(row, namespace=namespace) for row in rows]


def _row(row: tuple[Any, ...], *, namespace: str) -> dict[str, Any]:
    return {
        "event_id": row[0],
        "key": row[1],
        "value": row[2],
        "valid_from": row[3],
        "valid_to": row[4],
        "recorded_at": row[5],
        "status": row[6],
        "supersedes_id": row[7],
        "namespace": namespace,
    }