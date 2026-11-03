"""记忆层测试：三条设计各自要有牙——key 唯一、只追加、写前先提议。

这组测试用一个独立的 user 前缀，避免污染真实记忆。
"""

from __future__ import annotations

import uuid

import psycopg
import pytest

from recallgraph import memory
from recallgraph.storage import connect, ensure_schema, schema_ready


def _user() -> str:
    """每个测试一个独立 user：memory_events 禁止删除，靠隔离而不是靠清理。"""
    return f"pytest-{uuid.uuid4().hex[:10]}"


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
def user():
    return _user()


def test_add_then_recall(db, user) -> None:
    proposal = memory.propose(db, key="answer_style", value="先给结论，再给理由", user=user)
    assert proposal["operation"] == "add"
    assert proposal["previous_value"] is None
    assert memory.current(db, "answer_style", user=user) is None, "propose 绝不能写库"

    memory.commit(db, proposal)
    current = memory.current(db, "answer_style", user=user)
    assert current is not None
    assert current["value"] == "先给结论，再给理由"
    assert memory.recall(db, "结论", user=user)[0]["key"] == "answer_style"


def test_update_keeps_history(db, user) -> None:
    memory.commit(db, memory.propose(db, key="project_focus", value="RAG", user=user))
    memory.commit(db, memory.propose(db, key="project_focus", value="记忆评测", user=user))

    current = memory.current(db, "project_focus", user=user)
    assert current["value"] == "记忆评测"
    events = memory.history(db, "project_focus", user=user)
    assert [e["value"] for e in events] == ["RAG", "记忆评测"]
    assert events[0]["status"] == "superseded"
    assert events[1]["supersedes_id"] == events[0]["event_id"]
    assert events[1]["status"] == "active"


def test_same_value_is_noop(db, user) -> None:
    memory.commit(db, memory.propose(db, key="tone", value="直接一点", user=user))
    again = memory.propose(db, key="tone", value="直接一点", user=user)
    assert again["operation"] == "noop"
    assert memory.commit(db, again)["status"] == "noop"
    assert len(memory.history(db, "tone", user=user)) == 1


def test_retract_hides_but_keeps(db, user) -> None:
    memory.commit(db, memory.propose(db, key="temp", value="临时约定", user=user))
    memory.retract(db, "temp", user=user, reason="作废")
    assert memory.current(db, "temp", user=user) is None
    assert memory.recall(db, "临时", user=user) == []
    events = memory.history(db, "temp", user=user)
    assert [e["status"] for e in events] == ["retracted", "retracted"]
    assert events[1]["supersedes_id"] == events[0]["event_id"]


def test_as_of_time_travel(db, user) -> None:
    memory.commit(db, memory.propose(db, key="focus", value="RAG", user=user))
    memory.commit(db, memory.propose(db, key="focus", value="记忆评测", user=user))
    events = memory.history(db, "focus", user=user)
    first_ts, second_ts = events[0]["valid_from"], events[1]["valid_from"]

    at_first = memory.recall(db, "focus", user=user, as_of=first_ts)
    assert [item["value"] for item in at_first] == ["RAG"]
    at_second = memory.recall(db, "focus", user=user, as_of=second_ts)
    assert [item["value"] for item in at_second] == ["记忆评测"]


def test_one_active_per_key_is_enforced(db, user) -> None:
    memory.commit(db, memory.propose(db, key="dup", value="a", user=user))
    with pytest.raises(psycopg.errors.UniqueViolation):
        db.execute(
            """
            INSERT INTO memory_events (id, user_id, namespace, memory_key, operation, value, valid_from, status)
            VALUES ('01DUP', %s, 'user', 'dup', 'add', 'b', now(), 'active')
            """,
            (user,),
        )
    db.rollback()


def test_bad_key_or_value_rejected(db, user) -> None:
    with pytest.raises(memory.MemoryError):
        memory.propose(db, key="有空格 key", value="x", user=user)
    with pytest.raises(memory.MemoryError):
        memory.propose(db, key="ok", value="   ", user=user)


def test_commit_rejects_foreign_dict(db, user) -> None:
    with pytest.raises(memory.MemoryError):
        memory.commit(db, {"key": "x", "value": "y"})