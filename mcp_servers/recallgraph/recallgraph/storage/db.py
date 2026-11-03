"""数据库连接与建表。

设计上有意保持"薄"：连接串来自环境变量，建表由 sql/ 下的编号脚本驱动，
执行过的脚本记在 schema_migrations 里，重复执行不会报错。
"""

from __future__ import annotations

import os
from pathlib import Path

import psycopg

DEFAULT_DSN = "postgresql://recallgraph:recallgraph@127.0.0.1:5434/recallgraph"
SQL_DIR = Path(__file__).resolve().parents[1] / "sql"

_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""


def dsn() -> str:
    return os.environ.get("RECALLGRAPH_DSN", DEFAULT_DSN)


def connect(dsn_value: str | None = None) -> psycopg.Connection:
    return psycopg.connect(dsn_value or dsn(), connect_timeout=5)


def ensure_schema(conn: psycopg.Connection, dsn_value: str | None = None) -> list[str]:
    """按文件名顺序执行 sql/*.sql 中尚未执行的脚本，返回本次执行的版本号。"""
    conn.execute(_MIGRATIONS_TABLE)
    conn.commit()
    done = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}

    applied: list[str] = []
    for script in sorted(SQL_DIR.glob("*.sql")):
        version = script.stem
        if version in done:
            continue
        # 脚本自带 BEGIN/COMMIT，必须用 autocommit 连接执行，避免嵌套事务。
        with psycopg.connect(dsn_value or dsn(), autocommit=True) as raw:
            raw.execute(script.read_text(encoding="utf-8"))
        conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
        conn.commit()
        applied.append(version)
    return applied


def schema_ready(conn: psycopg.Connection) -> bool:
    row = conn.execute("SELECT to_regclass('public.chunks')").fetchone()
    return row is not None and row[0] is not None


def counts(conn: psycopg.Connection) -> dict[str, int]:
    result: dict[str, int] = {}
    for table in ("documents", "document_versions", "chunks", "entities", "claims", "evidence", "notes", "memory_events"):
        result[table] = int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
    return result