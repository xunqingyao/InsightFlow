"""笔记：人对库里结论留下的看法。

笔记和断言的差别是**作者不同**：断言来自论文（`asserted_by` 是文档版本），
笔记来自人（或 Agent 代笔）。笔记可以引用、赞同、反对某条断言，
所以它是"我的推测"这一侧的载体——检索时把两边并排，才能区分
"论文说了什么"和"我怎么想"。
"""

from __future__ import annotations

from typing import Any

import psycopg

from recallgraph.ids import new_ulid


def create_or_update_note(
    conn: psycopg.Connection,
    *,
    title: str,
    body: str,
    note_id: str | None = None,
    claim_ids: list[str] | tuple[str, ...] = (),
    version_ids: list[str] | tuple[str, ...] = (),
    author: str = "agent",
) -> dict[str, Any]:
    """新建或更新一条笔记，并把它引用到的断言挂上去。

    传 `note_id` 就是更新（保留原 id 与创建时间），不传就是新建。
    """
    title = (title or "").strip()
    body = (body or "").strip()
    if not title:
        raise ValueError("标题不能为空")
    if not body:
        raise ValueError("正文不能为空")

    if note_id:
        row = conn.execute("SELECT id FROM notes WHERE id = %s", (note_id,)).fetchone()
        if row is None:
            raise LookupError(f"笔记不存在：{note_id}")
        conn.execute(
            "UPDATE notes SET title = %s, body = %s, linked_version_ids = %s, updated_at = now() WHERE id = %s",
            (title, body, list(version_ids), note_id),
        )
    else:
        note_id = f"NOTE-{new_ulid()}"
        conn.execute(
            "INSERT INTO notes (id, author, title, body, linked_version_ids) VALUES (%s, %s, %s, %s, %s)",
            (note_id, author, title, body, list(version_ids)),
        )

    linked: list[str] = []
    missing: list[str] = []
    for claim_id in claim_ids:
        exists = conn.execute("SELECT 1 FROM claims WHERE id = %s", (claim_id,)).fetchone()
        if exists is None:
            missing.append(claim_id)
            continue
        conn.execute(
            "INSERT INTO note_claims (note_id, claim_id, relation) VALUES (%s, %s, 'cites') "
            "ON CONFLICT DO NOTHING",
            (note_id, claim_id),
        )
        linked.append(claim_id)

    return {
        "note_id": note_id,
        "title": title,
        "linked_claims": linked,
        "missing_claims": missing,
    }


def get_note(conn: psycopg.Connection, *, note_id: str) -> dict[str, Any] | None:
    """按 id 取一条笔记及其引用的断言。"""
    row = conn.execute(
        "SELECT id, author, title, body, linked_version_ids, created_at, updated_at "
        "FROM notes WHERE id = %s",
        (note_id,),
    ).fetchone()
    if row is None:
        return None
    claims = conn.execute(
        "SELECT c.id, c.claim_text, c.predicate FROM note_claims n JOIN claims c ON c.id = n.claim_id "
        "WHERE n.note_id = %s AND c.status = 'active' ORDER BY c.id",
        (note_id,),
    ).fetchall()
    return {
        "note_id": row[0],
        "author": row[1],
        "title": row[2],
        "body": row[3],
        "linked_version_ids": list(row[4]),
        "created_at": row[5].isoformat() if row[5] else None,
        "updated_at": row[6].isoformat() if row[6] else None,
        "claims": [{"claim_id": c[0], "claim": c[1], "predicate": c[2]} for c in claims],
    }