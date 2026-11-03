"""SQLite 持久化层。

设计取舍：
- 单文件 SQLite + WAL，够支撑单机演示与测试；不引入 ORM。
- 结构化字段以 JSON 列存储，读回时用 pydantic 重新校验，避免半结构化的脏数据流进报告。
- 每次调用独立连接（`with db.connect()`），避免跨线程共享连接。
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from insightflow.research.schemas import (
    Evidence,
    Finding,
    Report,
    RunCounters,
    RunRecord,
    RunStatus,
    ResearchRequest,
    Source,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    status       TEXT NOT NULL,
    request_json TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    report_id    TEXT,
    error        TEXT,
    stop_reason  TEXT,
    counters_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    source_id     TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    position      INTEGER NOT NULL DEFAULT 0,
    data_json     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sources_run ON sources(run_id, position);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    source_id   TEXT NOT NULL,
    data_json   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evidence_run ON evidence(run_id);

CREATE TABLE IF NOT EXISTS findings (
    finding_id  TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    report_id   TEXT NOT NULL,
    position    INTEGER NOT NULL DEFAULT 0,
    data_json   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_findings_report ON findings(report_id, position);

CREATE TABLE IF NOT EXISTS reports (
    report_id  TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    version    INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    data_json  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reports_run ON reports(run_id);

CREATE TABLE IF NOT EXISTS tool_calls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    tool        TEXT NOT NULL,
    ok          INTEGER NOT NULL,
    error_code  TEXT,
    started_at  TEXT NOT NULL,
    duration_ms INTEGER NOT NULL,
    request     TEXT,
    metadata    TEXT
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_run ON tool_calls(run_id);

CREATE TABLE IF NOT EXISTS run_events (
    run_id     TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    seq        INTEGER NOT NULL,
    type       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data_json  TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);

CREATE TABLE IF NOT EXISTS idempotency (
    key         TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    fingerprint TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""

# 结论先以草稿形式落库：MCP 工具是无状态的，每次调用都可能换一个进程，
# 只在内存里攒着，下一次 publish 就看不见了。
DRAFT_REPORT_ID = "__draft__"

_UPDATABLE_RUN_FIELDS = {"status", "updated_at", "report_id", "error", "stop_reason", "counters"}
_TERMINAL_STATUS_VALUES = ("completed", "partial", "failed", "cancelled")


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


class Database:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    # ---- runs -------------------------------------------------------------

    def create_run(self, record: RunRecord) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, status, request_json, created_at, updated_at,"
                " report_id, error, stop_reason, counters_json) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    record.run_id,
                    str(record.status),
                    record.request.model_dump_json(),
                    _iso(record.created_at),
                    _iso(record.updated_at),
                    record.report_id,
                    record.error,
                    record.stop_reason,
                    record.counters.model_dump_json(),
                ),
            )

    def update_run(
        self,
        run_id: str,
        *,
        expected: RunStatus | str | None = None,
        **fields: Any,
    ) -> bool:
        """更新运行字段；传入 `expected` 时只在状态仍为该值时生效。

        返回是否真的写入。长任务里“先查再做”与“取消”会竞争，状态迁移用条件更新
        才能保证取消不会被后到的 `completed` 覆盖。
        """
        unknown = set(fields) - _UPDATABLE_RUN_FIELDS
        if unknown:
            raise ValueError(f"不可更新的字段：{sorted(unknown)}")
        if not fields:
            return
        columns: list[str] = []
        values: list[Any] = []
        for key, value in fields.items():
            if key == "status":
                columns.append("status = ?")
                values.append(str(value))
            elif key == "counters":
                counters = value if isinstance(value, RunCounters) else RunCounters.model_validate(value)
                columns.append("counters_json = ?")
                values.append(counters.model_dump_json())
            elif key == "updated_at":
                columns.append("updated_at = ?")
                values.append(_iso(value))
            else:
                columns.append(f"{key} = ?")
                values.append(value)
        values.append(run_id)
        where = "run_id = ?"
        if expected is not None:
            where += " AND status = ?"
            values.append(str(expected))
        with self.connect() as conn:
            cursor = conn.execute(f"UPDATE runs SET {', '.join(columns)} WHERE {where}", values)
            return cursor.rowcount > 0

    def _row_to_run(self, row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            run_id=row["run_id"],
            status=RunStatus(row["status"]),
            request=ResearchRequest.model_validate_json(row["request_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            report_id=row["report_id"],
            error=row["error"],
            stop_reason=row["stop_reason"],
            counters=RunCounters.model_validate_json(row["counters_json"]),
        )

    def get_run(self, run_id: str) -> RunRecord | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._row_to_run(row) if row else None

    def delete_run(self, run_id: str) -> None:
        """删除一次运行及其全部派生物。只用于显式重跑（--force）。"""
        with self.connect() as conn:
            for table in ("findings", "reports", "evidence", "sources", "tool_calls"):
                conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))

    def list_runs(self, limit: int = 20) -> list[RunRecord]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def count_runs(self) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()
        return int(row["n"])

    # ---- sources / evidence ----------------------------------------------

    def replace_sources(self, run_id: str, sources: list[Source]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM sources WHERE run_id = ?", (run_id,))
            conn.executemany(
                "INSERT INTO sources (source_id, run_id, position, data_json) VALUES (?,?,?,?)",
                [
                    (item.source_id, run_id, index, item.model_dump_json())
                    for index, item in enumerate(sources)
                ],
            )

    def list_sources(self, run_id: str) -> list[Source]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT data_json FROM sources WHERE run_id = ? ORDER BY position", (run_id,)
            ).fetchall()
        return [Source.model_validate_json(row["data_json"]) for row in rows]

    def get_source(self, source_id: str) -> tuple[str, Source] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT run_id, data_json FROM sources WHERE source_id = ?", (source_id,)
            ).fetchone()
        if not row:
            return None
        return row["run_id"], Source.model_validate_json(row["data_json"])

    def add_evidence(self, run_id: str, items: list[Evidence]) -> None:
        if not items:
            return
        with self.connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO evidence (evidence_id, run_id, source_id, data_json)"
                " VALUES (?,?,?,?)",
                [
                    (item.evidence_id, run_id, item.source_id, item.model_dump_json())
                    for item in items
                ],
            )

    def list_evidence(self, run_id: str) -> list[Evidence]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT data_json FROM evidence WHERE run_id = ? ORDER BY evidence_id", (run_id,)
            ).fetchall()
        return [Evidence.model_validate_json(row["data_json"]) for row in rows]

    # ---- findings / reports ----------------------------------------------

    def replace_draft_findings(self, run_id: str, findings: list[Finding]) -> None:
        """把本次运行已收下的结论整批写进草稿区（publish 前可跨进程继续）。"""
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM findings WHERE run_id = ? AND report_id = ?",
                (run_id, DRAFT_REPORT_ID),
            )
            conn.executemany(
                "INSERT OR REPLACE INTO findings (finding_id, run_id, report_id, position, data_json)"
                " VALUES (?,?,?,?,?)",
                [
                    (item.finding_id, run_id, DRAFT_REPORT_ID, index, item.model_dump_json())
                    for index, item in enumerate(findings)
                ],
            )

    def list_draft_findings(self, run_id: str) -> list[Finding]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT data_json FROM findings WHERE run_id = ? AND report_id = ? ORDER BY position",
                (run_id, DRAFT_REPORT_ID),
            ).fetchall()
        return [Finding.model_validate_json(row["data_json"]) for row in rows]

    def save_report(self, report: Report, findings: list[Finding]) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO reports (report_id, run_id, version, created_at, data_json)"
                " VALUES (?,?,?,?,?)",
                (
                    report.report_id,
                    report.run_id,
                    report.version,
                    _iso(report.created_at),
                    report.model_dump_json(),
                ),
            )
            conn.execute("DELETE FROM findings WHERE report_id = ?", (report.report_id,))
            conn.execute(
                "DELETE FROM findings WHERE run_id = ? AND report_id = ?",
                (report.run_id, DRAFT_REPORT_ID),
            )
            conn.executemany(
                "INSERT INTO findings (finding_id, run_id, report_id, position, data_json)"
                " VALUES (?,?,?,?,?)",
                [
                    (
                        item.finding_id,
                        report.run_id,
                        report.report_id,
                        index,
                        item.model_dump_json(),
                    )
                    for index, item in enumerate(findings)
                ],
            )

    def get_report(self, report_id: str) -> Report | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT data_json FROM reports WHERE report_id = ?", (report_id,)
            ).fetchone()
        return Report.model_validate_json(row["data_json"]) if row else None

    def latest_report(self, run_id: str) -> Report | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT data_json FROM reports WHERE run_id = ? ORDER BY version DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        return Report.model_validate_json(row["data_json"]) if row else None

    # ---- tool call log ----------------------------------------------------

    def log_tool_call(
        self,
        run_id: str,
        tool: str,
        *,
        ok: bool,
        started_at: datetime,
        duration_ms: int,
        error_code: str | None = None,
        request: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO tool_calls (run_id, tool, ok, error_code, started_at, duration_ms,"
                " request, metadata) VALUES (?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    tool,
                    int(ok),
                    error_code,
                    _iso(started_at),
                    duration_ms,
                    json.dumps(request, ensure_ascii=False) if request else None,
                    json.dumps(metadata, ensure_ascii=False) if metadata else None,
                ),
            )

    def tool_call_stats(self, run_id: str) -> dict[str, int]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT tool, COUNT(*) AS n FROM tool_calls WHERE run_id = ? GROUP BY tool",
                (run_id,),
            ).fetchall()
        return {row["tool"]: int(row["n"]) for row in rows}
    # ---- events / idempotency --------------------------------------------

    def append_event(
        self,
        run_id: str,
        event_type: str,
        data: dict[str, Any] | None = None,
        *,
        only_if_active: bool = False,
    ) -> dict[str, Any] | None:
        """追加一条事件，`seq` 在同一 run 内单调递增。

        写入与取号在同一 `BEGIN IMMEDIATE` 事务里完成，避免两个线程拿到同一个 seq。
        `only_if_active=True` 时，若运行已是终态则丢弃该事件并返回 None——取消与
        “最后一个阶段事件”会竞争，状态判断必须和插入在同一个事务里，否则终态之后
        还会冒出 `plan_ready` 之类的事件，客户端无法判断什么时候可以停止等待。
        """
        created_at = datetime.now(UTC)
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if only_if_active:
                status_row = conn.execute(
                    "SELECT status FROM runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                if status_row is None or status_row["status"] in _TERMINAL_STATUS_VALUES:
                    return None
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS last FROM run_events WHERE run_id = ?", (run_id,)
            ).fetchone()
            seq = int(row["last"]) + 1
            payload = json.dumps(data or {}, ensure_ascii=False)
            conn.execute(
                "INSERT INTO run_events (run_id, seq, type, created_at, data_json)"
                " VALUES (?,?,?,?,?)",
                (run_id, seq, event_type, _iso(created_at), payload),
            )
        return {
            "run_id": run_id,
            "seq": seq,
            "type": event_type,
            "created_at": created_at.isoformat(),
            "data": data or {},
        }

    def list_events(
        self, run_id: str, *, after_seq: int = 0, limit: int = 500
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT seq, type, created_at, data_json FROM run_events"
                " WHERE run_id = ? AND seq > ? ORDER BY seq LIMIT ?",
                (run_id, after_seq, limit),
            ).fetchall()
        return [
            {
                "run_id": run_id,
                "seq": int(row["seq"]),
                "type": row["type"],
                "created_at": row["created_at"],
                "data": json.loads(row["data_json"]),
            }
            for row in rows
        ]

    def max_event_seq(self, run_id: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS last FROM run_events WHERE run_id = ?", (run_id,)
            ).fetchone()
        return int(row["last"])

    def request_cancel(self, run_id: str) -> bool:
        """把未终结的运行标记为 cancelled；返回是否真的发生了迁移。"""
        with self.connect() as conn:
            cursor = conn.execute(
                "UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ?"
                " AND status IN (?,?,?,?)",
                (
                    str(RunStatus.CANCELLED),
                    _iso(datetime.now(UTC)),
                    run_id,
                    str(RunStatus.QUEUED),
                    str(RunStatus.RUNNING),
                    str(RunStatus.WAITING_INPUT),
                    str(RunStatus.RECOVERABLE),
                ),
            )
            return cursor.rowcount > 0

    def save_idempotency(self, key: str, run_id: str, fingerprint: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO idempotency (key, run_id, fingerprint, created_at)"
                " VALUES (?,?,?,?)",
                (key, run_id, fingerprint, _iso(datetime.now(UTC))),
            )

    def get_idempotency(self, key: str) -> tuple[str, str] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT run_id, fingerprint FROM idempotency WHERE key = ?", (key,)
            ).fetchone()
        return (row["run_id"], row["fingerprint"]) if row else None
