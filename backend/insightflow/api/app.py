"""HTTP API。

同步执行 + 后台线程：一次运行耗时在秒级，先保证可观察、可重试。
过程用事件流暴露（`/events` 同一路径做内容协商：`Accept: text/event-stream`
走 SSE，否则返回 JSON 快照），客户端用 `Last-Event-ID` 断点续传。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, AsyncIterator

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from insightflow.api.render import render_markdown
from insightflow.config.settings import Settings, get_settings
from insightflow.research import events as event_types
from insightflow.research.budget import Budget
from insightflow.research.pipeline import ResearchEngine
from insightflow.research.schemas import (
    ResearchRequest,
    RunMode,
    RunRecord,
    RunStatus,
    Source,
    TaskType,
    stable_id,
)
from insightflow.research.time_window import normalize
from insightflow.storage.db import Database


class RunCreate(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    task_type: TaskType | None = None
    as_of: datetime | None = None
    days: int | None = Field(default=None, ge=1, le=365)
    timezone: str | None = None
    focus: list[str] | None = None
    mode: RunMode = RunMode.ONLINE
    fixture_id: str | None = None


class RunCreated(BaseModel):
    run_id: str
    status: RunStatus


def _request_from_payload(payload: RunCreate, settings: Settings) -> ResearchRequest:
    from insightflow.research.time_window import DEFAULT_TIMEZONE

    window = timedelta(days=payload.days) if payload.days else None
    return normalize(
        payload.query,
        as_of=payload.as_of,
        timezone=payload.timezone or settings.timezone or DEFAULT_TIMEZONE,
        window=window,
        task_type=payload.task_type,
        focus=payload.focus,
        mode=payload.mode,
        fixture_id=payload.fixture_id,
    )



def _recover_interrupted_runs(db: Database) -> int:
    """启动时把"上次进程留下的 running/queued"标成 recoverable。

    进程重启后不可能还有正在跑的线程，所以这些状态一定是中断的；标成
    `recoverable` 之后，`POST /runs/{id}/resume` 才能接手。
    """
    count = 0
    for record in db.list_runs(limit=500):
        if record.status in {RunStatus.RUNNING, RunStatus.QUEUED}:
            if db.update_run(
                record.run_id,
                expected=record.status,
                status=RunStatus.RECOVERABLE,
                stop_reason="process_restart",
                error="进程重启时中断，可用 resume 继续",
            ):
                db.append_event(
                    record.run_id,
                    "run_recoverable",
                    {"from": record.status.value, "reason": "process_restart"},
                    only_if_active=False,
                )
                count += 1
    return count


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    resolved.ensure_dirs()
    db = Database(resolved.db_path)
    db.init_schema()
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="research")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="InsightFlow", version="0.1.0", lifespan=lifespan)
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    _recover_interrupted_runs(db)
    app.state.settings = resolved
    app.state.db = db
    app.state.engine = ResearchEngine(resolved, db, budget=Budget())
    app.state.executor = executor

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html", media_type="text/html; charset=utf-8")

    @app.get("/api/health")
    def health(request: Request) -> dict[str, Any]:
        state = request.app.state
        return {
            "ok": True,
            "data": {
                "version": "0.1.0",
                "model_configured": state.settings.model_configured,
                "search_configured": state.settings.search_configured,
                "runs": state.db.count_runs(),
            },
            "error": None,
            "metadata": {"provider": "insightflow"},
        }

    @app.post("/api/research/runs", status_code=202, response_model=RunCreated)
    def create_run(payload: RunCreate, request: Request) -> RunCreated:
        state = request.app.state
        try:
            research_request = _request_from_payload(payload, state.settings)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        key = (request.headers.get("idempotency-key") or "").strip()
        fingerprint = _fingerprint(payload)
        if key:
            existing = state.db.get_idempotency(key)
            if existing is not None:
                run_id, seen = existing
                if seen != fingerprint:
                    raise HTTPException(
                        status_code=409, detail="同一 Idempotency-Key 不能用于不同的请求体"
                    )
                record = state.db.get_run(run_id)
                return RunCreated(run_id=run_id, status=record.status if record else RunStatus.QUEUED)
        now = datetime.now(tz=research_request.as_of.tzinfo)
        record = RunRecord(
            run_id=stable_id("run", research_request.query, research_request.as_of.isoformat()),
            status=RunStatus.QUEUED,
            request=research_request,
            created_at=now,
            updated_at=now,
        )
        if state.db.get_run(record.run_id):
            raise HTTPException(status_code=409, detail="同一问题与同一 as_of 的运行已存在")
        state.db.create_run(record)
        if key:
            state.db.save_idempotency(key, record.run_id, fingerprint)
        state.db.append_event(
            record.run_id,
            event_types.RUN_QUEUED,
            {
                "query": research_request.query,
                "mode": research_request.mode.value,
                "window_start": research_request.window_start.isoformat(),
                "window_end": research_request.window_end.isoformat(),
                "as_of": research_request.as_of.isoformat(),
            },
        )
        engine = state.engine
        state.executor.submit(engine.run, record.run_id)
        return RunCreated(run_id=record.run_id, status=record.status)

    @app.get("/api/research/runs")
    def list_runs(request: Request, limit: int = 20) -> dict[str, Any]:
        runs = request.app.state.db.list_runs(limit=min(max(limit, 1), 100))
        return {
            "ok": True,
            "data": [run.model_dump(mode="json") for run in runs],
            "error": None,
            "metadata": {"count": len(runs)},
        }

    @app.get("/api/research/runs/{run_id}")
    def get_run(run_id: str, request: Request) -> dict[str, Any]:
        state = request.app.state
        run = state.db.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="运行不存在")
        sources = state.db.list_sources(run_id)
        return {
            "ok": True,
            "data": {
                "run": run.model_dump(mode="json"),
                "sources": [source.model_dump(mode="json") for source in sources],
                "tool_calls": state.db.tool_call_stats(run_id),
            },
            "error": None,
            "metadata": {},
        }

    @app.get("/api/research/runs/{run_id}/report")
    def get_run_report(run_id: str, request: Request) -> dict[str, Any]:
        state = request.app.state
        run = state.db.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="运行不存在")
        report = state.db.latest_report(run_id)
        if report is None:
            raise HTTPException(status_code=404, detail="报告尚未生成")
        return {
            "ok": True,
            "data": _report_payload(report),
            "error": None,
            "metadata": {},
        }

    @app.get("/api/fixtures")
    def list_fixtures(request: Request) -> dict[str, Any]:
        fixtures = _list_fixtures(request.app.state.settings.fixtures_dir)
        return {"ok": True, "data": fixtures, "error": None, "metadata": {"count": len(fixtures)}}

    @app.get("/api/reports/{report_id}")
    def get_report(report_id: str, request: Request) -> dict[str, Any]:
        report = request.app.state.db.get_report(report_id)
        if report is None:
            raise HTTPException(status_code=404, detail="报告不存在")
        return {
            "ok": True,
            "data": _report_payload(report),
            "error": None,
            "metadata": {},
        }

    @app.get("/api/sources/{source_id}")
    def get_source(source_id: str, request: Request) -> dict[str, Any]:
        found = request.app.state.db.get_source(source_id)
        if found is None:
            raise HTTPException(status_code=404, detail="来源不存在")
        run_id, source = found
        return {
            "ok": True,
            "data": {"run_id": run_id, "source": source.model_dump(mode="json")},
            "error": None,
            "metadata": {},
        }

    @app.get("/api/research/runs/{run_id}/events")
    async def run_events(
        run_id: str, request: Request, after: int = 0, limit: int = 500
    ) -> Any:
        state = request.app.state
        if state.db.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="运行不存在")
        cursor = max(after, 0)
        if "text/event-stream" in request.headers.get("accept", ""):
            last_event_id = (request.headers.get("last-event-id") or "").strip()
            if last_event_id.isdigit():
                cursor = max(cursor, int(last_event_id))
            return StreamingResponse(
                _event_stream(state.db, run_id, cursor),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        rows = state.db.list_events(run_id, after_seq=cursor, limit=min(max(limit, 1), 1000))
        return {
            "ok": True,
            "data": rows,
            "error": None,
            "metadata": {
                "next_after": rows[-1]["seq"] if rows else cursor,
                "max_seq": state.db.max_event_seq(run_id),
            },
        }

    @app.post("/api/research/runs/{run_id}/resume", status_code=202)
    def resume_run(run_id: str, request: Request) -> dict[str, Any]:
        state = request.app.state
        record = state.db.get_run(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="运行不存在")
        if record.status not in {RunStatus.RECOVERABLE, RunStatus.QUEUED}:
            raise HTTPException(
                status_code=409,
                detail=f"当前状态是 {record.status.value}，只有 recoverable/queued 可以续跑",
            )
        if not state.db.update_run(run_id, expected=record.status, status=RunStatus.QUEUED):
            raise HTTPException(status_code=409, detail="另一个进程已经接手这个运行")
        counters = record.counters.model_copy(update={"resumes": record.counters.resumes + 1})
        state.db.update_run(run_id, counters=counters)
        state.executor.submit(state.engine.run, run_id)
        return {
            "ok": True,
            "data": {"run_id": run_id, "status": RunStatus.QUEUED.value},
            "error": None,
            "metadata": {"resumed_from": record.status.value, "resumes": counters.resumes},
        }

    @app.post("/api/research/runs/{run_id}/cancel", status_code=202)
    def cancel_run(run_id: str, request: Request) -> dict[str, Any]:
        state = request.app.state
        record = state.db.get_run(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="运行不存在")
        if record.status in TERMINAL_STATUSES:
            raise HTTPException(status_code=409, detail=f"运行已结束（{record.status}）")
        if not state.db.request_cancel(run_id):
            raise HTTPException(status_code=409, detail="运行状态已变化，取消失败")
        state.db.append_event(
            run_id,
            event_types.RUN_CANCELLED,
            {"status": str(RunStatus.CANCELLED), "reason": "user_requested"},
        )
        current = state.db.get_run(run_id)
        return {
            "ok": True,
            "data": {
                "run_id": run_id,
                "status": str(current.status if current else RunStatus.CANCELLED),
            },
            "error": None,
            "metadata": {},
        }

    return app


TERMINAL_STATUSES = frozenset(
    {
        RunStatus.COMPLETED,
        RunStatus.PARTIAL,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    }
)
SSE_POLL_SECONDS = 0.2
SSE_PING_SECONDS = 15.0
SSE_BATCH = 200


def _fingerprint(payload: RunCreate) -> str:
    raw = json.dumps(payload.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _format_sse(event: dict[str, Any]) -> str:
    """SSE 帧：只带 id 与 data，事件类型放在 data.type 里。

    不用 `event:` 字段，客户端一个 message 监听器就能收全部类型；
    重连时 `Last-Event-ID` 即上一条的 id。
    """
    payload = json.dumps(event, ensure_ascii=False)
    return f"id: {event['seq']}\ndata: {payload}\n\n"


async def _event_stream(db: Database, run_id: str, cursor: int) -> AsyncIterator[str]:
    """先补历史（`seq > cursor`），再按 200ms 轮询增量；终态事件发出后关闭。

    SSE 重连语义：客户端带上 `Last-Event-ID`，只收到该 id 之后的事件，
    因此重复投递不会重复处理（`seq` 单调）。
    """
    idle = 0.0
    while True:
        rows = await anyio.to_thread.run_sync(
            lambda: db.list_events(run_id, after_seq=cursor, limit=SSE_BATCH)
        )
        if rows:
            idle = 0.0
            for event in rows:
                cursor = event["seq"]
                yield _format_sse(event)
                if event["type"] in event_types.TERMINAL_EVENTS:
                    return
            continue
        record = await anyio.to_thread.run_sync(lambda: db.get_run(run_id))
        if record is None:
            return
        if record.status in TERMINAL_STATUSES:
            # 终态但没有新事件：可能终态事件已在上一轮读走，也可能写入失败；都该断开。
            pending = await anyio.to_thread.run_sync(
                lambda: db.list_events(run_id, after_seq=cursor, limit=SSE_BATCH)
            )
            if not pending:
                return
            continue
        idle += SSE_POLL_SECONDS
        if idle >= SSE_PING_SECONDS:
            idle = 0.0
            yield ": ping\n\n"
        await asyncio.sleep(SSE_POLL_SECONDS)


def _report_payload(report: Any) -> dict[str, Any]:
    """Markdown 给命令行与下载，HTML 给网页；两者都来自同一份报告正文。"""
    return {
        "report": report.model_dump(mode="json"),
        "markdown": report.markdown,
        "html": render_markdown(report.markdown),
    }


def _list_fixtures(fixtures_dir: Path) -> list[dict[str, Any]]:
    """列出可用的回放资料，供界面下拉框使用；坏掉的清单跳过而不是整体失败。"""
    if not fixtures_dir.is_dir():
        return []
    found: list[dict[str, Any]] = []
    for path in sorted(fixtures_dir.iterdir()):
        manifest = path / "manifest.json"
        if not manifest.is_file():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        items = data.get("items") or []
        found.append(
            {
                "id": str(data.get("fixture_id") or path.name),
                "description": data.get("description", ""),
                "synthetic": bool(data.get("synthetic", True)),
                "items": len(items),
            }
        )
    return found


def default_source_payload(source: Source) -> dict[str, Any]:
    return source.model_dump(mode="json")


app = create_app()